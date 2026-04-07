import os
import click
import logging

from importlib.metadata import version, PackageNotFoundError
from pharmacophore2mol.logging_config import setup_logging

try:
    __version__ = version("pharmacophore2mol")
except PackageNotFoundError:
    # Package not installed, mark as development version
    __version__ = "dev"

logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version=__version__, prog_name='pharmacophore2mol')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output (DEBUG level)')
@click.option('-q', '--quiet', is_flag=True, help='Suppress all output except errors')
@click.pass_context
def cli(ctx, verbose, quiet):
    """
    Pharmacophore2Mol: Command Line Interface (CLI)
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store verbosity settings in context
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    
    # Initialize logging
    setup_logging(verbose=verbose, quiet=quiet)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=True, readable=True))
def evaluate(input_path):
    """
    Evaluate molecules from SDF or XYZ file(s).
    
    Automatically converts XYZ to SDF using OpenBabel if needed.
    
    Examples:
    
        >>> p2m evaluate xyz_dir
        >>> p2m evaluate molecules.sdf

    """
    from .metrics.utils import evaluate_from_file
    
    logger.debug(f"Starting evaluation for: {input_path}")
    
    results = evaluate_from_file(input_path)
    results.print_summary()
    
    logger.debug("Evaluation complete")


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=True, readable=True))
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--keep-disconnected', is_flag=True, help='Keep molecules with disconnected fragments')
@click.option('--keep-unstable', is_flag=True, help='Ignore molecular stability checks')
def clean(input_path, output_path, keep_disconnected, keep_unstable):
    """
    Clean molecule dataset by removing invalid molecules.
    
    Removes molecules that fail rdkit parsing or have deeper structural issues (valence, formal charges, disconnected components).
    Consolidates multiple input files into a single output SDF.
    Supports SDF, XYZ, MOL2, PDB formats (single or multi-file).
    
    Examples:
    
        p2m clean molecules.sdf cleaned.sdf
        
        p2m clean some/dir/with/files/ cleaned.sdf

        p2m clean molecules.sdf cleaned.sdf --keep-disconnected
    """
    from pharmacophore2mol.data.preprocessing import clean_molecules

    logger.debug(f"Starting cleaning: {input_path} to {output_path}")

    num_valid = clean_molecules(
        input_path=input_path,
        output_path=output_path,
        remove_disconnected=not keep_disconnected,
        only_stable=not keep_unstable
    )

    logger.info(f"Cleaning complete: {num_valid} valid molecules saved")


@cli.command(name="tb")
@click.argument('logdir', required=False, default=None)
def tensorboard_cmd(logdir):
    """
    Launch TensorBoard.
    
    If LOGDIR is provided, it first checks if it's a subfolder in the `runs` directory.
    If not found there, it treats the input as a direct filesystem path.
    If LOGDIR is omitted, it defaults to the `runs` directory.
    
    Examples:
    
        p2m tb                # Opens runs/
        p2m tb bonder         # Opens runs/bonder/
        p2m tb ./custom_runs  # Opens ./custom_runs/
    """
    import sys
    import subprocess
    from pathlib import Path
    from pharmacophore2mol import BASE_DIR
    
    runs_dir = BASE_DIR / "runs"
    
    if not logdir:
        target_dir = runs_dir
    else:
        run_subfolder = runs_dir / logdir
        if run_subfolder.is_dir(): #found in runs/
            target_dir = run_subfolder
        else: #must be a direct path
            target_path = Path(logdir)
            if target_path.is_dir():
                target_dir = target_path
            else:
                click.echo(f"Error: Could not find log directory for '{logdir}'", err=True)
                sys.exit(1)

    click.echo(f"Starting TensorBoard with logdir: {target_dir}")
    try:
        # sys.executable ensures we use the same python environment
        subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir", str(target_dir)])
    except KeyboardInterrupt:
        click.echo("\nTensorBoard stopped.")
