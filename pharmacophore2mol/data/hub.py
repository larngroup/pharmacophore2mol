import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import pharmacophore2mol as p2m

def download_if_missing(filename: str, repo_id: str = None) -> Path:
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
