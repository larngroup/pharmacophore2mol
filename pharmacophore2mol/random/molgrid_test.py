import torch
import molgrid
# import molgrid.openbabel as pybel
from rdkit import Chem
import os


if __name__ == "__main__":

    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    # Load multiple molecules from an SDF file
    suppl = Chem.SDMolSupplier("../data/zinc3d_test.sdf")

    # Convert to a list and filter out None (failed molecules)
    molecules = [mol for mol in suppl if mol is not None]

    print(f"Loaded {len(molecules)} molecules")

    # Print SMILES for the first few molecules
    for i, mol in enumerate(molecules[:5]):
        print(f"Molecule {i+1}: {Chem.MolToSmiles(mol)}")
    print("...")

    mol_block = Chem.MolToMolBlock(molecules[0])
    print(mol_block)
    c = molgrid.CoordinateSet(mol_block)
    print(c.coords.shape)







