import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import pharmacophore2mol as p2m

def get_dataset(filename: str, repo_id: str = None) -> Path:
    """
    Ensures a single file is available locally in the data/raw directory, 
    downloading it from the Hugging Face Hub if it is missing.
    """
    if repo_id is None:
        repo_id = p2m.HF_REPO

    local_dir = p2m.RAW_DATA_DIR
    local_dir.mkdir(parents=True, exist_ok=True)
    
    actual_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir #enforces a symlink or copy
    )

    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"Failed to download {filename} from Hugging Face Hub.")
    
    return Path(actual_path)


def ensure_dataset_file(filepath, repo_id: str = None) -> Path:
    """
    Resolve a dataset file path. If the file is missing and points to the
    data/raw folder, download it from the Hugging Face Hub.
    """
    path = Path(filepath)
    if path.exists():
        return path

    is_raw_path = (
        path.parent == p2m.RAW_DATA_DIR
        or path.parent.name == "raw"
        or ("data" in path.parts and "raw" in path.parts)
    )

    if is_raw_path and path.suffix == ".sdf":
        return get_dataset(path.name, repo_id=repo_id)

    return path
