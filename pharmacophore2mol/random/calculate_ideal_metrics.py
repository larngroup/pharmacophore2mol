import sys
import argparse
from pathlib import Path

# Add project root to path so we can import from pharmacophore2mol
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pharmacophore2mol.metrics.utils import evaluate_from_file

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check to calculate 'ideal' metrics on a ground truth dataset (e.g., your training SDF)."
    )
    parser.add_argument(
        "input_sdf", 
        type=str, 
        nargs="?", 
        # default=str(project_root / "dump" / "train_5confs.sdf"), # fallback default, can be changed
        help="Path to the original/ground-truth SDF file."
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_sdf)
    if not input_path.exists():
        print(f"Error: Could not find file {input_path}")
        print("Please provide a valid path to an SDF dataset of real molecules.")
        sys.exit(1)
        
    print(f"=== Running Sanity Check on {input_path.name} ===")
    print("Calculating quality metrics for real molecules.")
    print("Ideally, metrics like Valid Valency (Atom/Mol Stability) and Validity should be exactly 100%.\n")
    
    results = evaluate_from_file(str(input_path))
    
    # print_summary automatically outputs a clean table
    results.print_summary()

if __name__ == "__main__":
    main()
