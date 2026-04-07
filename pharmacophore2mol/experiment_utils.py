from datetime import datetime
import heapq
import inspect
import shutil
import subprocess
import sys
import os
import re
import textwrap
import importlib
import importlib.util
from coolname import generate_slug
from pathlib import Path

GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"
BLUE = "\033[1;34m"
RED = "\033[1;31m"


class _OrderedLogWriter:
    """Single ordered writer for stdout+stderr.

    Each write/update is assigned a strictly increasing sequence number.
    Pending lines are flushed to disk in sequence order, ensuring deterministic
    interleaving across both streams.

    If `timestamp` is enabled, each flushed line is prefixed with a timestamp.
    If `flush_age_seconds` is provided, lines older than that age are flushed
    automatically regardless of line position.
    """
    def __init__(
        self,
        log_file,
        timestamp=False,
        timestamp_format="%d/%m/%y %H:%M:%S",
        flush_age_seconds=None,
    ):
        self.log_file = log_file
        self.next_seq = 0
        self.pending = {}  # key -> (seq, ts, text)
        self.heap = []     # min-heap of (seq, key)
        self.timestamp = timestamp
        self.timestamp_format = timestamp_format
        self.flush_age_seconds = flush_age_seconds

    def _alloc_seq(self):
        seq = self.next_seq
        self.next_seq += 1
        return seq

    def update(self, key, text):
        """Register/update a line.

        The latest text for a key is kept, and the seq order is tracked.
        """
        seq = self._alloc_seq()
        ts = datetime.now() if self.timestamp else None
        self.pending[key] = (seq, ts, text)
        heapq.heappush(self.heap, (seq, key))
        return seq

    def flush_up_to(self, seq_limit):
        """Flush all pending entries with seq <= seq_limit."""
        while self.heap and self.heap[0][0] <= seq_limit:
            seq, key = heapq.heappop(self.heap)
            v = self.pending.get(key)
            if v is None:
                continue
            if v[0] != seq:
                continue
            _, ts, text = v
            if self.timestamp and ts is not None:
                ts_str = ts.strftime(self.timestamp_format)
                self.log_file.write(f"[{ts_str}] {text.rstrip()}\n")
            else:
                self.log_file.write(text.rstrip() + "\n")
            del self.pending[key]

    def flush_old_by_age(self):
        """Flush lines whose last update is older than flush_age_seconds."""
        if not self.timestamp or self.flush_age_seconds is None:
            return

        now = datetime.now()
        while self.heap:
            seq, key = self.heap[0]
            v = self.pending.get(key)
            if v is None:
                heapq.heappop(self.heap)
                continue

            if v[0] != seq:
                heapq.heappop(self.heap)
                continue

            _, ts, _ = v
            if ts is None:
                break

            if (now - ts).total_seconds() >= self.flush_age_seconds:
                # Flush everything up to this seq (it is the oldest timestamped entry).
                self.flush_up_to(seq)
                continue

            break

    def flush_all(self):
        """Flush everything pending."""
        if self.heap:
            self.flush_up_to(self.next_seq - 1)
        self.log_file.flush()


class _Tee:
    """Wrapper for stdout/stderr that emits ordered log output.

    Each line is tracked and updated in a shared ordered writer so stdout and
    stderr content is merged deterministically.
    """
    def __init__(self, original_stream, mux, stream_name):
        self.original_stream = original_stream
        self._mux = mux
        self._stream_name = stream_name

        # Virtual terminal state
        self.lines = [""]
        self.line_seq = [0]
        self.cy = 0
        self.cx = 0
        self.flushed_y = 0

        # Match \r, \n, \b, and CSI-style escape sequences (like \x1b[A, \x1b[2K, \x1b[36m, \x1b[?25h)
        self._ansi_pattern = re.compile(r'(\x1b\[[0-9;?]*[a-zA-Z]|\r|\n|\b)')
        # Secondary strip for any leftover generic ANSI codes
        self._strip_ansi = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, data):
        self.original_stream.write(data)

        if isinstance(data, bytes):
            try:
                data = data.decode('utf-8')
            except UnicodeDecodeError:
                data = data.decode('utf-8', errors='replace')

        data = data.replace('\r\n', '\n')
        tokens = self._ansi_pattern.split(data)
        
        for token in tokens:
            if not token:
                continue
            if token == '\n':
                self.cy += 1
                self.cx = 0
                while len(self.lines) <= self.cy:
                    self.lines.append("")
                    self.line_seq.append(0)
            elif token == '\r':
                self.cx = 0
            elif token == '\b':
                self.cx = max(0, self.cx - 1)
            elif token.startswith('\x1b['):
                char = token[-1]
                args = token[2:-1].replace('?', '')
                if char == 'A': # Cursor Up
                    n = int(args) if args else 1
                    self.cy = max(0, self.cy - n)
                elif char == 'B': # Cursor Down
                    n = int(args) if args else 1
                    self.cy += n
                    while len(self.lines) <= self.cy:
                        self.lines.append("")
                        self.line_seq.append(0)
                elif char == 'C': # Cursor Forward
                    n = int(args) if args else 1
                    self.cx += n
                elif char == 'D': # Cursor Back
                    n = int(args) if args else 1
                    self.cx = max(0, self.cx - n)
                elif char == 'K': # Erase in Line
                    n = int(args) if args else 0
                    line = self.lines[self.cy]
                    if n == 0:
                        self.lines[self.cy] = line[:self.cx]
                    elif n == 1:
                        self.lines[self.cy] = " " * self.cx + line[self.cx:]
                    elif n == 2:
                        self.lines[self.cy] = ""
            else:
                token = self._strip_ansi.sub('', token)
                if not token:
                    continue
                line = self.lines[self.cy]
                if len(line) < self.cx:
                    line += " " * (self.cx - len(line))
                self.lines[self.cy] = line[:self.cx] + token + line[self.cx + len(token):]
                self.cx += len(token)

            # Update the global ordered writer with our latest line state.
            key = (self._stream_name, self.cy)
            self.line_seq[self.cy] = self._mux.update(key, self.lines[self.cy])

            # Flush based on age when enabled, else fall back to the old line-count heuristic.
            if self._mux.flush_age_seconds is not None:
                self._mux.flush_old_by_age()
            else:
                while self.flushed_y < self.cy - 100:
                    if self.lines[self.flushed_y] is not None:
                        seq_to_flush = self.line_seq[self.flushed_y]
                        self._mux.flush_up_to(seq_to_flush)
                        self.lines[self.flushed_y] = None
                    self.flushed_y += 1
                
        return len(data)

    def _full_flush(self):
        self.original_stream.flush()

        # Flush remaining virtual terminal lines through the ordered writer.
        # This ensures stdout/stderr ordering is preserved in the log.
        self._mux.flush_all()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


class save_run:
    """
    Context manager to save the current run's environment and any relevant information for reproducibility.
    It generates a unique slug for each run and creates a folder to store the run's data.
    The resulting folder will be organized as follows:
        
        runs/
        ├── 260317-172004_bold-badger/      <-- Run ID
        |   ├── console.log                 <-- Captured stdout/stderr
        |   ├── git_commit.txt              <-- Stable commit hash (if available)
        |   ├── uncommitted_changes.patch    <-- Patch of uncommitted changes (if any)
        |   ├── weights/                    <-- Model checkpoints
        |   ├── code_snapshot/              <-- Plaintext repo structure (.py only by default)
        |   ├── restore_code_snapshot.py     <-- Restore repo files from this snapshot (requires consent)
        |   |
        |   | [other things can be added by the user]
        |   |
        |   ├── events.out.tfevents...      <-- TensorBoard logs
        |   ├── images/                     <-- Generated images, training curves...
        |   └── ...                         <-- Any other relevant files
        
    Usage
    -----
    ```python
    with save_run() as run:
        # Save checkpoints using the automatically created weights directory
        torch.save(model.state_dict(), run.weights_dir / "model_epoch_1.pt")
        
        # Save custom files anywhere inside the run's base directory
        plt.savefig(run.run_dir / "images" / "loss_curve.png")
    ```

    Parameters
    ----------
    base_dir : str
        Base directory to store runs (default: "runs").
    run_name : str | None
        Optional fixed run name. If None, a timestamp+slug is generated.
    dry_run : bool
        If True, writes to `.dry_run` and clears any existing folder there.
    timestamp : bool
        Whether to prefix log lines with timestamps.
    timestamp_format : str
        strftime format string used for timestamps.
    flush_every : float | None
        Flush log lines older than this many seconds (defaults to 10.0). If None,
        falls back to the legacy line-count flushing heuristic.

    """

    def __init__(
        self,
        base_dir="runs",
        run_name=None,
        dry_run=False,
        timestamp=True,
        timestamp_format="%d-%m-%y %H:%M:%S",
        flush_every=10.0,
    ):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.timestamp = timestamp
        self.timestamp_format = timestamp_format
        self.flush_age_seconds = flush_every
        if self.dry_run:
            self.run_name = ".dry_run"#to be able to debug easily
        else:
            self.run_name = run_name or self._generate_run_name()

        self.run_dir = self.base_dir / self.run_name
        self.weights_dir = self.run_dir / "weights"
        self.code_snapshot_dir = self.run_dir / "code_snapshot"
        # self.images_dir = self.run_dir / "images"
        self.log_path = self.run_dir / "console.log"

        self._orig_streams = (None, None)
        self._log_file = None

    def __enter__(self):
        if self.dry_run:
            if self.run_dir.exists():
                shutil.rmtree(self.run_dir)
            
            print(YELLOW + "!" * 50)
            print("!!! DRY RUN ACTIVE: Saving to .dry_run sandbox")
            print("!!! Existing sandbox data has been cleared.")
            print("!!! Use dry_run=False to save runs normally.")
            print("!" * 50 + RESET)

        
        
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.weights_dir.mkdir()
        self.code_snapshot_dir.mkdir()
        # self.images_dir.mkdir()

        # os.environ["FORCE_COLOR"] = "1"
        self._orig_streams = (sys.stdout, sys.stderr)

        # Flush before swapping underlying handles
        sys.stdout.flush()
        sys.stderr.flush()

        self._log_file = open(self.log_path, "a", encoding="utf-8", buffering=1) #line buffered

        # Record git state (if applicable)
        self._record_git_state()

        # Shared ordered writer ensures stdout+stderr are merged in print order.
        self._mux = _OrderedLogWriter(
            self._log_file,
            timestamp=self.timestamp,
            timestamp_format=self.timestamp_format,
            flush_age_seconds=self.flush_age_seconds,
        )

        sys.stdout = _Tee(sys.stdout, self._mux, "stdout")
        sys.stderr = _Tee(sys.stderr, self._mux, "stderr")

        self._take_code_snapshot()

        # Add helper scripts to restore the codebase from this snapshot
        self._create_restore_snapshot_scripts()

        self.start_time = datetime.now()

        return self
    
    def _generate_run_name(self):
        slug = generate_slug(2)
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        return f"{timestamp}_{slug}"

    def _record_git_state(self):
        """Record git commit hash + uncommitted changes if git is present."""
        try:
            # Determine git repo root (if any)
            git_root = None
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0:
                    git_root = Path(res.stdout.strip())
            except Exception:
                git_root = None

            if git_root is None or not git_root.exists():
                return

            # Commit hash
            commit = None
            try:
                res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=git_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0:
                    commit = res.stdout.strip()
            except Exception:
                commit = None

            if commit:
                with open(self.run_dir / "git_commit.txt", "w", encoding="utf-8") as f:
                    f.write(commit + "\n")

            # We capture a patch containing unstaged and staged changes (if any).
            # This is the authoritative snapshot of any work-in-progress state.

            # Create a patch capturing the current uncommitted diff (including staged changes)
            try:
                patch_parts = []
                for args in (["git", "diff"], ["git", "diff", "--cached"]):
                    res = subprocess.run(
                        args,
                        cwd=git_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        patch_parts.append(res.stdout)

                if patch_parts:
                    with open(self.run_dir / "uncommitted_changes.patch", "w", encoding="utf-8") as f:
                        f.write("\n".join(patch_parts))
            except Exception:
                pass

        except Exception:
            # Fail silently if git isn't available or something goes wrong
            pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
            sys.stderr.flush()

        #final message to indiocate finish via __exit__, with no crashes
        duration = datetime.now() - self.start_time
        timestamp = datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S')
        
        success = exc_type is None
        main_tag = f"{GREEN}[OK]" if success else f"{RED}[!!]"
        main_msg = "RUN COMPLETED SUCCESSFULLY." if success else "RUN COMPLETED WITH EXCEPTIONS."
        
        if self.dry_run:
            s_tag = f"{YELLOW}[!!]"
            s_suffix = f" (will be overwritten on next dry run){RESET}"
        else:
            s_tag = f"{GREEN}[OK]"
            s_suffix = f"{RESET}"
        
        dir_str = str(self.run_dir)

        print(f"\n{GREEN if success else RED}{'*' * 68}")
        print(f"{main_tag}  {main_msg}")
        print(f"{GREEN}[OK]  NO HARDCRASHES, OR THIS MESSAGE WOULD BE ABSENT.")
        
        print(f"{s_tag}  Storage: {dir_str}{s_suffix}")
        
        print(f"{BLUE}[IN]  System time: {timestamp}")
        print(f"{BLUE}[IN]  Total run duration: {duration}")
        print(f"{GREEN if success else RED}{'*' * 68}{RESET}")

        # Flush everything before destroying FDs
        sys.stdout._full_flush()
        sys.stderr._full_flush()

        sys.stdout, sys.stderr = self._orig_streams

        if self._log_file is not None:
            self._log_file.close()

        if self.dry_run:
            return False

        return False #this allows exceptions to propagate normally
    
    def _take_code_snapshot(self):
        """Recursively copies source code and config files to the snapshot directory.
        Also ignores common large or sensitive files and directories to avoid issues with storage limits and sensitive data leaks.
        Uses os.walk() with in-place pruning to avoid OS-level symlink crashes.
        """
        src_root = Path.cwd()
        
        # Directories to completely ignore
        ignore_dirs = {self.base_dir.name, 'venv', 'env', '__pycache__', 'data'}
        
        # File extensions that are safe and necessary to backup
        allowed_extensions = {
            '.py',              # Source code
            '.yaml', '.yml',    # Configs
            '.json', '.toml',   # Configs / Project metadata
            '.txt',             # requirements.txt
            '.sh',              # Bash launch scripts
            '.md'               # READMEs
        }

        filesize_limit_bytes = 5 * 1024 * 1024 # 5 MB

        for root, dirs, files in os.walk(src_root):
            # Ignore unwanted directories (including the run output folder).
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ignore_dirs]

            root_path = Path(root)
            for file_name in files:
                #ignoring hidden files and folders for safety
                if file_name.startswith('.'):
                    continue
                    
                file_path = root_path / file_name
                
                # Check extension whitelist
                if file_path.suffix not in allowed_extensions:
                    continue

                try:
                    if file_path.stat().st_size > filesize_limit_bytes:
                        continue
                except OSError:
                    continue
                    
                try:
                    rel_path = file_path.relative_to(src_root)
                    dest_path = self.code_snapshot_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                except OSError:# skip files that cannot be copied (locked, etc.).
                    continue

    def _create_restore_snapshot_scripts(self):
        """Generate a self-contained, cross-platform restore script.

        The script overwrites any repo files that match the snapshot. It requires
        typing the exact run name to proceed.
        """
        run_dir = self.run_dir
        script_path = run_dir / "restore_code_snapshot.py"
        run_name = run_dir.name

        def _restore_code_snapshot_script_template():
            # This function exists purely to allow using inspect.getsource().
            # The generated script is the body of this function.

            #!/usr/bin/env python3

            import shutil
            import subprocess
            from pathlib import Path

            RUN_NAME = '<RUN_NAME>'

            def _repo_root() -> Path:
                try:
                    res = subprocess.run(
                        ['git', 'rev-parse', '--show-toplevel'],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        return Path(res.stdout.strip())
                except Exception:
                    pass

                return Path(__file__).resolve().parents[2]

            def main() -> int:
                root = _repo_root()
                run_dir = Path(__file__).resolve().parent

                RED = "\033[1;31m"
                RESET = "\033[0m"
                print(RED + "DANGER:" + RESET + " This will overwrite files in your repository that share the same paths as the snapshot.")
                print(RED + "Run name:" + RESET, RUN_NAME)
                confirm = input('If you understand, please type the name of this run to proceed: ').strip()
                if confirm != RUN_NAME:
                    print('Aborted: run name did not match. No changes were made.')
                    return 1

                snap = run_dir / 'code_snapshot'
                if not snap.is_dir():
                    print('Snapshot not found:', snap)
                    return 2

                files = [p for p in snap.rglob('*') if p.is_file()]
                print('Restoring', len(files), 'files from snapshot...')

                for src in files:
                    rel = src.relative_to(snap)
                    dst = root / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

                print('Restore complete.')
                return 0

            if __name__ == '__main__':
                raise SystemExit(main())

        try:
            src = inspect.getsource(_restore_code_snapshot_script_template)
            # just drop the `def ...` line and dedent the rest.
            src = "\n".join(src.splitlines()[1:])
            src = textwrap.dedent(src)

            # Keep only from the shebang onward.
            lines = src.splitlines()
            for i, line in enumerate(lines):
                if line.startswith('#!'):
                    lines = lines[i:]
                    break
            src = "\n".join(lines)

            src = src.replace("'<RUN_NAME>'", repr(run_name))

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(src)
            try:
                os.chmod(script_path, 0o755)
            except Exception:
                pass
        except Exception:
            pass




if __name__ == "__main__":
    from tqdm import tqdm
    import time

    # Example usage of the save_run context manager
    # Enable timestamped log entries by setting timestamp=True
    with save_run(dry_run=True, flush_every=0.5) as run:
        print(f"Run directory created at: {run.run_dir}")
        print(f"Weights will be saved to: {run.weights_dir}")
        print(f"Code snapshots will be saved to: {run.code_snapshot_dir}")
    
    
    # if True:
        print("Starting a tqdm progress bar test (checks buffering)...")
        for i in tqdm(range(50), desc="Testing tqdm"):
            if i == 25:
                print("Halfway tqdm")
            time.sleep(0.1)
            
        print("Starting a rich progress bar test...")
        from rich.progress import Progress
        with Progress() as progress:
            task = progress.add_task("[cyan]Testing rich...", total=500)
            while not progress.finished:
                progress.advance(task)
                time.sleep(0.001)
                if progress.tasks[0].completed == 25:
                    print("Halfway rich")

        print("This is a test log message. It should appear in the console and be saved to console.log.")
        
        # #not a hard crash, but simulating an unhandled exception
        # sys.exit(1)

        #simulate a hard crash (like a segfault)
        # print("Now, we will segfault the main process, so this should be the last process originated message on the log")
        # import os
        # os._exit(1)
        # print("This one should not appear anywhere!")

        # This file is primarily an import-time library.
    # The block below is just a demo showing how to use save_run/load_run.


class _CodeProxy:
    """Dynamic wiring that turns dot access into an import.

    This proxy drives `old_run.code.<module>...` syntax. It delegates to the
    owning `load_run` instance so that imports are sourced from the snapshot
    and not from the current live workspace.
    """

    def __init__(self, loader, base=""):
        self._loader = loader
        self._base = base

    def __getattr__(self, module_name):
        candidate = f"{self._base}.{module_name}" if self._base else module_name

        # If the snapshot contains an actual module for this path, resolve it.
        # This prevents falling back to the live codebase when the live package
        # exists but the snapshot only has a deeper file structure.
        if self._loader._snapshot_path_for_module(candidate) is not None:
            return self._loader.import_module(candidate)

        # Otherwise, keep building the dotted path until we hit an actual module.
        return _CodeProxy(self._loader, candidate)


class load_run:
    """Context manager for safely importing historical code from a run snapshot.

    This context manager patches:
      1) sys.path (so the snapshot is preferred for imports)
      2) sys.modules (hides the current live code while the context is active)

    The goal is to let you evaluate old code side-by-side with current code
    without contaminating the running process.
    """

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.snapshot_dir = self.run_dir / "code_snapshot"
        self.weights_dir = self.run_dir / "weights"

        if not self.snapshot_dir.is_dir():
            raise FileNotFoundError(f"[CHRONICLE] No code snapshot found at {self.snapshot_dir}")

        self._live_modules_backup = {}
        # Normalize for consistent comparisons on Windows (case-insensitive file system)
        self._live_cwd = os.path.normcase(str(Path.cwd().resolve()))

        # nice convenience for dot-notation access (requires __init__.py in packages)
        self.code = _CodeProxy(self)

    def _snapshot_path_for_module(self, module_name: str) -> Path | None:
        """Return the filesystem path to a module inside the snapshot (if it exists)."""
        parts = module_name.split('.')

        candidate = self.snapshot_dir.joinpath(*parts).with_suffix('.py')
        if candidate.is_file():
            return candidate

        candidate = self.snapshot_dir.joinpath(*parts, '__init__.py')
        if candidate.is_file():
            return candidate

        return None

    def __enter__(self):
        # 1) Patch sys.path so the snapshot is searched first
        sys.path.insert(0, str(self.snapshot_dir))

        # 2) Cache-swap live modules so they don't shadow snapshot modules
        live_root = self._live_cwd
        for name, mod in list(sys.modules.items()):
            if not hasattr(mod, "__file__") or not mod.__file__:
                continue
            mod_path = os.path.normcase(os.path.abspath(str(mod.__file__)))
            if mod_path.startswith(live_root) and ".venv" not in mod_path and "venv" not in mod_path and "site-packages" not in mod_path:
                self._live_modules_backup[name] = mod
                del sys.modules[name]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1) Remove snapshot from sys.path
        try:
            sys.path.remove(str(self.snapshot_dir))
        except ValueError:
            pass

        # 2) Remove any modules loaded from the snapshot
        snapshot_prefix = os.path.normcase(str(self.snapshot_dir))
        for name, mod in list(sys.modules.items()):
            if not hasattr(mod, "__file__") or not mod.__file__:
                continue
            mod_path = os.path.normcase(os.path.abspath(str(mod.__file__)))
            if mod_path.startswith(snapshot_prefix):
                del sys.modules[name]

        # 3) Restore the live modules
        sys.modules.update(self._live_modules_backup)

        return False  # propagate exceptions normally

    def import_module(self, path: str):
        """Import a module by string path.

        This will first attempt a normal import. If the result comes from the
        live codebase (i.e., the module file is outside the snapshot), it will
        fall back to importing the exact .py file from the snapshot directory.
        """
        module = importlib.import_module(path)

        # If the imported module is not from the snapshot, try directly loading
        # it from the snapshot file (useful when the snapshot lacks __init__.py).
        if hasattr(module, "__file__") and module.__file__:
            mod_path = os.path.normcase(os.path.abspath(str(module.__file__)))
            snapshot_root = os.path.normcase(str(self.snapshot_dir))
            if not mod_path.startswith(snapshot_root):
                snapshot_file = self.snapshot_dir.joinpath(*path.split("."))
                if snapshot_file.with_suffix(".py").is_file():
                    snapshot_file = snapshot_file.with_suffix(".py")
                elif (snapshot_file / "__init__.py").is_file():
                    snapshot_file = snapshot_file / "__init__.py"
                else:
                    return module

                spec = importlib.util.spec_from_file_location(path, snapshot_file)
                new_mod = importlib.util.module_from_spec(spec)
                sys.modules[path] = new_mod
                if spec.loader is None:
                    raise ImportError(f"Cannot load module {path} from {snapshot_file}")
                spec.loader.exec_module(new_mod)
                return new_mod

        return module

    def get_best_weights(self):
        """Return the best weight file from the snapshot's weights folder."""
        best_weights = list(self.weights_dir.glob("*best*.pt"))
        if best_weights:
            return best_weights[0]

        all_weights = list(self.weights_dir.glob("*.pt"))
        if all_weights:
            return all_weights[-1]

        raise FileNotFoundError(f"No .pt files found in {self.weights_dir}")


if __name__ == "__main__":
    # Demo: instantiate MyModel from the CURRENT codebase and two historical runs.

    from snapshot_me.snapshot_me import MyModel# as LiveModel

    # live = LiveModel()
    live = MyModel()
    print("Live model param:", live.param)

    for run_dir in [
        Path("runs/260319-182017_beryl-goshawk"),
        Path("runs/260319-182726_adaptable-koel"),
    ]:
        with load_run(run_dir) as old_run:
            # Option A: bulletproof import (works even without __init__.py)
            old_mod = old_run.import_module("snapshot_me.snapshot_me")
            old_model = old_mod.MyModel()
            print(f"Run {run_dir.name} param:", old_model.param)

            # Option B: using the proxy object (requires __init__.py in package)
            # Option B: using the proxy object (works even when the snapshot
            # doesn't have __init__.py; it will resolve the deepest module file)
            old_model2 = old_run.code.snapshot_me.snapshot_me.MyModel()
            print(f"Run {run_dir.name} param (proxy):", old_model2.param)

    live = MyModel()
    print("Live model param after loading old runs:", live.param)
    print("Done demo")
