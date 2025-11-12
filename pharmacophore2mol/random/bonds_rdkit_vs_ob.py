"""
This is just to test which of the two libraries (RDKit or OpenBabel) produces better bonds when faced with increasing coordinate noise.
If RDKit is better, we can just use it for bond perception in noisy scenarios, and no extra dependencies are needed.
Also, this checks times for both libraries.
"""

from rdkit import Chem




def add_coordinate_noise(mol, noise_level):
    """
    Adds Gaussian noise to the atomic coordinates of a molecule.
    
    Args:
        mol: RDKit Mol object with 3D coordinates
        noise_level: Standard deviation of the Gaussian noise to add (in Angstroms)
    Returns:
        New RDKit Mol object with noisy coordinates
    """
    import numpy as np
    from rdkit.Chem import AllChem

    noisy_mol = Chem.Mol(mol)
    conf = noisy_mol.GetConformer()
    
    for atom_idx in range(noisy_mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(atom_idx))
        noise = np.random.normal(0, noise_level, size=3)
        new_pos = pos + noise
        conf.SetAtomPosition(atom_idx, new_pos.tolist())
    
    return noisy_mol


def compare_bonds(mol_ref, mol_inferred):
    """
    Compares the bonds of two molecules and returns the number of matching bonds.
    
    Args:
        mol_ref: Reference RDKit Mol object
        mol_inferred: Inferred RDKit Mol object

    Returns:
        int: Number of matching bonds
    """
    bond_ref = set()
    bond_inferred = set()

    for bond in mol_ref.GetBonds():
        bond_ref.add((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()))

    for bond in mol_inferred.GetBonds():
        bond_inferred.add((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()))

    return len(bond_ref.intersection(bond_inferred))


def write_mol_to_sdf(mol, filepath):
    """
    Writes an RDKit Mol object to an SDF file.
    
    Args:
        mol: RDKit Mol object
        filepath: Path to output SDF file
    """
    writer = Chem.SDWriter(filepath)
    writer.write(mol)
    writer.close()


if __name__ == "__main__":
    from pharmacophore2mol.data.utils import CustomSDMolSupplier



    mol_supplier = CustomSDMolSupplier("./dump/train_5confs.sdf")
    data = []
    count = 0
    for mol in mol_supplier:
        if mol is not None:
            data.append(mol)
            count += 1
        if count >= 1:
            break

    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]  # in Angstroms
    write_mol_to_sdf(data[0], "./dump/original.sdf")
    write_mol_to_sdf(add_coordinate_noise(data[0], 0.25), "./dump/noisy_0.25.sdf")
    write_mol_to_sdf(add_coordinate_noise(data[0], 0.5), "./dump/noisy_0.5.sdf")

    for noise in noise_levels:
        print(f"Noise level: {noise} Å")
        for mol in data:
            noisy_mol = add_coordinate_noise(mol, noise)
            print(compare_bonds(mol, noisy_mol))
            