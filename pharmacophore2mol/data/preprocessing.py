from pharmacophore2mol.data.utils import CustomSDMolSupplier
from rdkit import Chem
# from pharmacophore2mol.metrics.evaluator import



def clean_molecules(input_path: str, output_path: str, keep_disconnected: bool = False, keep_unstable: bool = False) -> int:
    """
    Clean molecule dataset by removing invalid molecules.
    
    Removes molecules that fail sanitization or have structural issues.
    
    Args:
        input_path: Path to input SDF
        output_path: Path to output SDF file
        keep_disconnected: Whether to keep molecules with disconnected fragments
        keep_unstable: Whether to ignore molecular stability checks
    Returns:
        Number of valid molecules saved
    """

    mol_supplier = CustomSDMolSupplier(input_path)
    writer = Chem.SDWriter(output_path)
    num_valid = 0

    for mol in mol_supplier:
        ...
    
