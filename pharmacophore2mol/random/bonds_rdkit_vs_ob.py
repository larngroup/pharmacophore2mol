"""
This is just to test which of the two libraries (RDKit or OpenBabel) produces better bonds when faced with increasing coordinate noise.
If RDKit is better, we can just use it for bond perception in noisy scenarios, and no extra dependencies are needed.
Also, this checks times for both libraries.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
from pharmacophore2mol.data.utils import SANITIZE_DEFAULT_OPS, suppress_openbabel_warnings, suppress_rdkit_warnings
try:
    from openbabel import pybel
except Exception as e:
    raise ImportError(
        "OpenBabel 'pybel' is required for add_bonds_from_coords_openbabel. "
        "Install with 'pip install openbabel openbabel-wheel' or use your system package manager."
    ) from e


def add_coordinate_noise(mol, noise_level):
    """
    Adds Gaussian noise to the atomic coordinates of a molecule.
    
    Args:
        mol: RDKit Mol object with 3D coordinates
        noise_level: Standard deviation of the Gaussian noise to add (in Angstroms)
    Returns:
        New RDKit Mol object with noisy coordinates
    """


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
    if mol_ref is None or mol_inferred is None:
        return 0.0
    if mol_ref.GetNumAtoms() != mol_inferred.GetNumAtoms():
        return 0.0  # Different number of atoms, cannot compare bonds
    if mol_ref.GetNumBonds() == 0:
        return 1.0 if mol_inferred.GetNumBonds() == 0 else 0.0
    
    bond_ref = set()
    bond_inferred = set()

    for bond in mol_ref.GetBonds():
        bond_ref.add((*tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))), bond.GetBondType()))

    for bond in mol_inferred.GetBonds():
        bond_inferred.add((*tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))), bond.GetBondType()))


    misses = bond_ref - bond_inferred
    extras = bond_inferred - bond_ref
    eps = 1e-8
    return 1 - (len(misses) / (len(bond_ref) + eps))


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

def reorder_atoms_by_coords(mol_ref, mol_shuffled, tol=1e-3):
    """Return a copy of mol_shuffled reordered to match mol_ref by coordinates."""
    conf_ref, conf_shuf = mol_ref.GetConformer(), mol_shuffled.GetConformer()
    mapping = {}
    for i, a_ref in enumerate(mol_ref.GetAtoms()):
        pos_ref = np.array(conf_ref.GetAtomPosition(i))
        for j, a_sh in enumerate(mol_shuffled.GetAtoms()):
            pos_sh = np.array(conf_shuf.GetAtomPosition(j))
            if (a_ref.GetSymbol() == a_sh.GetSymbol() and
                np.linalg.norm(pos_ref - pos_sh) < tol):
                mapping[j] = i
                break
    if len(mapping) != mol_ref.GetNumAtoms():
        raise ValueError("Could not map all atoms — mismatch in coords or elements.")

    # Reorder
    new_order = [j for j, i in sorted(mapping.items(), key=lambda kv: kv[1])]
    return Chem.RenumberAtoms(mol_shuffled, new_order)

def add_bonds_from_coords_openbabel(mol: Chem.Mol) -> Chem.Mol:
    """
    Use Open Babel to infer bonds from 3D coordinates of an RDKit Mol.
    Uses in-memory XYZ string instead of temporary files.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have 3D coordinates (a conformer).")
    conf = mol.GetConformer()

    # xyz str (kinda monkey ngl)
    n_atoms = mol.GetNumAtoms()
    xyz_lines = [f"{n_atoms}", ""]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    xyz_str = "\n".join(xyz_lines)
    with suppress_openbabel_warnings():
        obmol = pybel.readstring("xyz", xyz_str)
        obmol.OBMol.ConnectTheDots()
        obmol.OBMol.PerceiveBondOrders()

        mol_block = obmol.write("mol")  # MOL is smaller than SDF for this use case
    
    with suppress_rdkit_warnings():
        mol_out = Chem.MolFromMolBlock(mol_block, sanitize=False)
        
        try:
            Chem.SanitizeMol(mol_out, Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except:
            pass

        # mol_out = reorder_atoms_by_coords(mol, mol_out) #seems to not be needed
    
    return mol_out


def add_bonds_from_coords_rdkit(mol: Chem.Mol, use_hueckel: bool = False, charge: int = 0, 
                                 cov_factor: float = 1.3, allow_charged_fragments: bool = True,
                                 embed_chiral: bool = True, use_vdw: bool = False) -> Chem.Mol:
    """
    Use RDKit's rdDetermineBonds to infer bonds from 3D coordinates.
    
    This is a pure RDKit solution that doesn't require OpenBabel.
    
    Args:
        mol: RDKit Mol object with 3D coordinates (a conformer)
        use_hueckel: If True, use extended Hueckel theory for connectivity
                     (more accurate but slower). If False, use van der Waals or
                     connect-the-dots method (faster)
        charge: Molecular charge (required for Hueckel method if non-zero)
        cov_factor: Factor to multiply covalent radii (for van der Waals method)
        allow_charged_fragments: If True, assign formal charges based on valency;
                                 otherwise use radical electrons
        embed_chiral: If True, embed chirality information (calls sanitizeMol)
        use_vdw: If False, use connect-the-dots; if True, use van der Waals method
    
    Returns:
        RDKit Mol object with bonds determined from coordinates
    """
    
    with suppress_rdkit_warnings():
        
        if mol.GetNumConformers() == 0:
            raise ValueError("Molecule must have 3D coordinates (a conformer).")
        
        # Create a copy to avoid modifying the original
        mol_copy = Chem.Mol(mol)
        
        # Remove all existing bonds
        mol_copy = Chem.RWMol(mol_copy)
        bonds_to_remove = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) 
                        for bond in mol_copy.GetBonds()]
        for begin, end in reversed(bonds_to_remove):
            mol_copy.RemoveBond(begin, end)
        mol_copy = mol_copy.GetMol()
        
        # Determine bonds from coordinates
        try:
            rdDetermineBonds.DetermineBonds(
                mol_copy,
                useHueckel=use_hueckel,
                charge=charge,
                covFactor=cov_factor,
                allowChargedFragments=allow_charged_fragments,
                embedChiral=embed_chiral,
                useAtomMap=False,
                useVdw=use_vdw
            )
        except (ValueError, RuntimeError) as e:
            # If bond determination fails, return None or try with different parameters
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Bond determination failed: {e}")
            return None
        # mol_copy = reorder_atoms_by_coords(mol, mol_copy) #seems to not be needed
            
    return mol_copy


def print_atoms_in_order(mol):
    atom_list = [atom for atom in mol.GetAtoms()]
    atom_list = sorted(atom_list, key=lambda atom: (atom.GetIdx()))
    atom_list = [f"{atom.GetSymbol()}" for atom in atom_list]
    print("".join(atom_list))

if __name__ == "__main__":
    import logging
    
    # Set logging level to control warning suppression
    # Use logging.DEBUG to see all warnings (verbose mode)
    # Use logging.INFO or logging.WARNING to suppress warnings (normal mode)
    # logging.basicConfig(level=logging.DEBUG)  # Change to logging.DEBUG for verbose
    # logging.basicConfig(level=logging.WARNING)  # Change to logging.DEBUG for verbose
    
    from pharmacophore2mol.data.utils import CustomSDMolSupplier



    mol_supplier = CustomSDMolSupplier("./dump/train_5confs.sdf")
    data = []
    count = 0
    for mol in mol_supplier:
        if mol is not None:
            data.append(mol)
            count += 1
        if count >= 1000:
            break

    noise_levels = np.linspace(0.0, 0.7, 50)  # in Angstroms
    write_mol_to_sdf(data[0], "./dump/original.sdf")
    write_mol_to_sdf(add_coordinate_noise(data[0], 0.25), "./dump/noisy_0.25.sdf")
    write_mol_to_sdf(add_coordinate_noise(data[0], 0.5), "./dump/noisy_0.5.sdf")

    # Store results for each noise level
    results = []
    
    # Calculate total iterations for progress bar
    total_iterations = len(noise_levels) * len(data)
    
    with tqdm(total=total_iterations, unit="mol") as pbar:
        for noise in noise_levels:
            # Accumulate scores for this noise level
            ob_scores = []
            rdkit_dots_scores = []
            rdkit_vdw_scores = []
            rdkit_hueckel_scores = []
            
            for mol in data:
                noisy_mol = add_coordinate_noise(mol, noise)
                noisy_mol_ob = add_bonds_from_coords_openbabel(noisy_mol)
                
                # Test RDKit method (default: connect-the-dots)
                noisy_mol_rdkit_dots = add_bonds_from_coords_rdkit(noisy_mol, use_vdw=False)
                
                # Test RDKit method (van der Waals)
                noisy_mol_rdkit_vdw = add_bonds_from_coords_rdkit(noisy_mol, use_vdw=True)

                # Test RDKit method (Hueckel)
                noisy_mol_rdkit_hueckel = add_bonds_from_coords_rdkit(
                    noisy_mol, use_hueckel=True, charge=Chem.GetFormalCharge(mol)
                )
                
                # Calculate bond matches
                ob_scores.append(compare_bonds(mol, noisy_mol_ob))
                rdkit_dots_scores.append(compare_bonds(mol, noisy_mol_rdkit_dots))
                rdkit_vdw_scores.append(compare_bonds(mol, noisy_mol_rdkit_vdw))
                rdkit_hueckel_scores.append(compare_bonds(mol, noisy_mol_rdkit_hueckel))
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Noise": f"{noise:.2f} Å"})
            
            # Calculate averages for this noise level
            results.append([
                f"{noise:.2f}",
                f"{np.mean(ob_scores):.4f}",
                f"{np.mean(rdkit_dots_scores):.4f}",
                f"{np.mean(rdkit_vdw_scores):.4f}",
                f"{np.mean(rdkit_hueckel_scores):.4f}"
            ])
    
    # Print results table
    headers = ["Noise (Å)", "OpenBabel", "RDKit (dots)", "RDKit (vdW)", "RDKit (Hueckel)"]
    print("\n" + "="*80)
    print("Bond Perception Accuracy vs Coordinate Noise")
    print("="*80)
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("="*80)
    
    # Create matplotlib plot
    # Extract data from results
    noise_values = [float(row[0]) for row in results]
    ob_values = [float(row[1]) for row in results]
    rdkit_dots_values = [float(row[2]) for row in results]
    rdkit_vdw_values = [float(row[3]) for row in results]
    rdkit_hueckel_values = [float(row[4]) for row in results]
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(noise_values, ob_values, linewidth=2, markersize=8, label='OpenBabel', color='#2E86AB')
    plt.plot(noise_values, rdkit_dots_values, linewidth=2, markersize=8, label='RDKit (connect-the-dots)', color='#A23B72')
    plt.plot(noise_values, rdkit_vdw_values, linewidth=2, markersize=8, label='RDKit (van der Waals)', color='#F18F01')
    plt.plot(noise_values, rdkit_hueckel_values, linewidth=2, markersize=8, label='RDKit (Hueckel)', color='#C73E1D')
    
    plt.xlabel('Coordinate Noise (Å)', fontsize=12, fontweight='bold')
    plt.ylabel('Bond Matching Accuracy', fontsize=12, fontweight='bold')
    plt.title('Bond Perception Methods: Accuracy vs Coordinate Noise', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(noise_values[0], noise_values[-1])
    plt.ylim(-0.05, 1.05)
    
    # Add horizontal line at perfect accuracy
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1, label='Perfect accuracy')
    
    plt.tight_layout()
    plt.show()
    
    print("\n Plot displayed!")
