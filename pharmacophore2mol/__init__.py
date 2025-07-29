import os
from pathlib import Path

# Base project directory (parent of this __init__.py file)
BASE_DIR = Path(__file__).parent.parent
PROJECT_DIR = Path(__file__).parent  # pharmacophore2mol package directory

# Data directories
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
# CACHE_DIR = DATA_DIR / "cache"

# Model directories
MODELS_DIR = PROJECT_DIR / "models"
