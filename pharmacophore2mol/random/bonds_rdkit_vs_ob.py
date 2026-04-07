"""
This is just to test which of the two libraries (RDKit or OpenBabel) produces better bonds when faced with increasing coordinate noise.
If RDKit is better, we can just use it for bond perception in noisy scenarios, and no extra dependencies are needed.
Also, this checks times for both libraries.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import random

from pharmacophore2mol.data.utils import SANITIZE_DEFAULT_OPS, suppress_openbabel_warnings, suppress_rdkit_warnings
from pharmacophore2mol.experiment_utils import load_run
from pharmacophore2mol import BASE_DIR

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

def add_bonds_from_coords_bonder(mol: Chem.Mol, model, device) -> Chem.Mol:
    """Uses a PyTorch Bonder network to predict bonds from RDKit Mol coordinates."""
    if mol.GetNumConformers() == 0:
        return mol

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return mol
        
    coords = []
    atomic_numbers = []
    
    conf = mol.GetConformer()
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        atomic_numbers.append(mol.GetAtomWithIdx(i).GetAtomicNum())
        
    coords_t = torch.tensor([coords], dtype=torch.float32).to(device)
    atomic_numbers_t = torch.tensor([atomic_numbers], dtype=torch.long).to(device)
    atom_mask_t = torch.ones((1, num_atoms), dtype=torch.bool).to(device)
    
    with torch.no_grad():
        logits, _ = model(coords_t, atomic_numbers_t, atom_mask_t)
        predictions = torch.argmax(logits, dim=-1)[0]
        
    new_mol = Chem.RWMol(mol)
    # Remove existing bonds
    bonds_to_remove = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in new_mol.GetBonds()]
    for begin, end in reversed(bonds_to_remove):
        new_mol.RemoveBond(begin, end)
        
    # Standard mapping 
    bond_type_mapping = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC
    }
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms): # Only check upper triangle
            pred_order = int(predictions[i, j].item())
            if pred_order > 0 and pred_order in bond_type_mapping:
                new_mol.AddBond(i, j, bond_type_mapping[pred_order])
                
    with suppress_rdkit_warnings():
        try:
            Chem.SanitizeMol(new_mol, Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except:
            pass

    return new_mol.GetMol()

if __name__ == "__main__":
    import logging
    from pharmacophore2mol.data.utils import CustomSDMolSupplier

    ATOM_CAP = 80  #skip molecules with more than this many atoms to avoid OOM or long processing times
    # Define your bonder models here
    # "Legend Name" : Configuration
    BONDER_MODELS = {
        # "Bonder (sigma=0)": {
        #     "run_dir": "runs/bonder/260405-120144_exotic-unicorn", 
        #     "weights": None,  # filename in weights folder, or None to auto-pick best
        #     "color": "#ffe600"  # Yellow/Orange
        # },
        "Bonder qm9 (sigma=0.1)": {
            "run_dir": "runs/bonder/260405-143855_cuddly-skunk", 
            "weights": None,
            "color": "#ffa722"  # Orange/Red
        },
            "Bonder geom (sigma=0.1)": {
            "run_dir": "runs/bonder/260406-110031_lavender-kiwi", 
            "weights": None,
            "color": "#ff5222"  # Orange/Red
        },
        # "Bonder (sigma=0.3)": {
        #     "run_dir": "runs/bonder/260405-160329_khaki-centipede", 
        #     "weights": None,
        #     "color": "#cc7700"  # Deep Red
        # },
        # "Bonder (sigma=0.5)": {
        #     "run_dir": "runs/bonder/260406-035639_astonishing-coua", 
        #     "weights": None,
        #     "color": "#cc0000ff"
        # }
    }

    # mol_supplier = CustomSDMolSupplier("./dump/train_5confs.sdf")
    # mol_supplier = CustomSDMolSupplier("./pharmacophore2mol/data/raw/qm9_test.sdf")
    mol_supplier = CustomSDMolSupplier("./pharmacophore2mol/data/raw/geom_5confs_test.sdf")
    # mol_supplier = CustomSDMolSupplier("./pharmacophore2mol/data/raw/zinc3d_test.sdf")
    data = []
    count = 0
    rng = random.Random(42)  # For reproducibility of molecule selection
    picking_sequence = list(range(len(mol_supplier)))
    rng.shuffle(picking_sequence)
    for i in picking_sequence:
        mol = mol_supplier[i]
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            if num_atoms > ATOM_CAP:
                continue
            data.append(mol)
            count += 1
        if count >= 1000:
            break

    noise_levels = np.linspace(0.0, 0.7, 50)  # in Angstroms
    write_mol_to_sdf(data[0], "./dump/original.sdf")
    write_mol_to_sdf(add_coordinate_noise(data[0], 0.25), "./dump/noisy_0.25.sdf")
    write_mol_to_sdf(add_coordinate_noise(data[0], 0.5), "./dump/noisy_0.5.sdf")

    total_iters = len(noise_levels) * len(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    methods = {
        'OpenBabel': {'color': '#1f77b4', 'func': lambda m, orig: add_bonds_from_coords_openbabel(m)}, # Blue
        'RDKit (connect-the-dots)': {'color': '#008080', 'func': lambda m, orig: add_bonds_from_coords_rdkit(m, use_vdw=False)}, # Teal
        'RDKit (van der Waals)': {'color': '#9467bd', 'func': lambda m, orig: add_bonds_from_coords_rdkit(m, use_vdw=True)}, # Purple
        'RDKit (Hueckel)': {'color': '#17becf', 'func': lambda m, orig: add_bonds_from_coords_rdkit(m, use_hueckel=True, charge=Chem.GetFormalCharge(orig))}, # Cyan
    }

    # Data structure to hold all scores
    # results_data["Method Name"] = [score_for_noise_0, score_for_noise_1, ...]
    results_data = {}

    # 1. EVALUATE CLASSICAL METHODS
    for method_name, config in methods.items():
        print(f"\nEvaluating {method_name}...")
        func = config['func']
        
        # Reset random seed per method to ensure identical noisy coordinates
        np.random.seed(42)
        
        scores_per_noise = []
        with tqdm(total=total_iters, desc=method_name, unit="mol") as pbar:
            for noise in noise_levels:
                noise_scores = []
                for orig_mol in data:
                    noisy_mol = add_coordinate_noise(orig_mol, noise)
                    inferred_mol = func(noisy_mol, orig_mol)

                    score = compare_bonds(orig_mol, inferred_mol)
                    noise_scores.append(score)
                    pbar.update(1)
                scores_per_noise.append(np.mean(noise_scores))
                pbar.set_postfix({"Noise": f"{noise:.2f} Å"})
        
        results_data[method_name] = scores_per_noise

    # 2. EVALUATE BONDER MODELS
    for model_name, cfg in BONDER_MODELS.items():
        print(f"\nEvaluating Bonder Model: {model_name}...")
        
        run_dir_path = Path(BASE_DIR) / cfg["run_dir"]
        weights_name = cfg.get("weights")
        
        train_module = None
        
        # Load the run context safely
        try:
            with load_run(run_dir_path) as loaded_code:
                train_module = loaded_code.import_module("pharmacophore2mol.models.bonder.train")
                run_config = getattr(train_module, "CONFIG", None)
                DenseBondPredictor = train_module.DenseBondPredictor
                if weights_name is None:
                    weights_path = loaded_code.get_best_weights()
                else:
                    weights_path = loaded_code.weights_dir / weights_name
        except Exception:
            pass

        if train_module is None:
            # Fallback to the current live code if the snapshot failed to save the 'bonder' folder
            from pharmacophore2mol.models.bonder.train import CONFIG as run_config, DenseBondPredictor
            from pharmacophore2mol.experiment_utils import load_run
            
            # Since importing from the snapshot failed, we extract weights using load_run directly
            loader = load_run(run_dir_path)
            if weights_name is None:
                weights_path = loader.get_best_weights()
            else:
                weights_path = loader.weights_dir / weights_name
            
        # Setup kwargs dynamically using the config
        model_kwargs = {
            "num_atom_types": run_config["max_atomic_number"] + 1,
            "atom_embedding_dim": run_config["atom_embedding_dim"],
            "hidden_dim": run_config["hidden_dim"],
            "num_layers": run_config["num_layers"],
            "cutoff": run_config["distance_cutoff"],
            "min_distance": run_config.get("min_distance", 0.1),
            "num_classes": 5
        }
        
        # Init model
        model = DenseBondPredictor(**model_kwargs)
        
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        # Reset random seed per method to ensure identical noisy coordinates
        np.random.seed(42)
        
        scores_per_noise = []
        with tqdm(total=total_iters, desc=model_name, unit="mol") as pbar:
            for noise in noise_levels:
                noise_scores = []
                for orig_mol in data:
                    noisy_mol = add_coordinate_noise(orig_mol, noise)
                    inferred_mol = add_bonds_from_coords_bonder(noisy_mol, model, device)
                    score = compare_bonds(orig_mol, inferred_mol)
                    noise_scores.append(score)
                    pbar.update(1)
                scores_per_noise.append(np.mean(noise_scores))
                pbar.set_postfix({"Noise": f"{noise:.2f} Å"})
        
        results_data[model_name] = scores_per_noise

    # 3. PRINT RESULTS TABLE
    headers = ["Noise (Å)"] + list(methods.keys()) + list(BONDER_MODELS.keys())
    
    table_rows = []
    for i, noise in enumerate(noise_levels):
        row = [f"{noise:.2f}"]
        for name in headers[1:]:
            row.append(f"{results_data[name][i]:.4f}")
        table_rows.append(row)
        
    print("\n" + "="*80)
    print(f"Bond Perception Accuracy vs Coordinate Noise (Capped at {ATOM_CAP} atoms)")
    print("="*80)
    print(tabulate(table_rows, headers=headers, tablefmt="grid"))
    print("="*80)
    
    # 4. PLOT RESULTS
    plt.figure(figsize=(12, 7))
    
    # Plot classical
    for method_name, config in methods.items():
        plt.plot(noise_levels, results_data[method_name], linewidth=2, markersize=8, label=method_name, color=config['color'])
        
    # Plot NN models
    for model_name, cfg in BONDER_MODELS.items():
        color = cfg.get("color", "#33a02c")
        plt.plot(noise_levels, results_data[model_name], linewidth=2, markersize=8, label=model_name, color=color)
    
    plt.xlabel('Coordinate Noise (Å)', fontsize=12, fontweight='bold')
    plt.ylabel('Bond Matching Accuracy', fontsize=12, fontweight='bold')
    plt.title('Bond Perception Methods: Accuracy vs Coordinate Noise', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(noise_levels[0], noise_levels[-1])
    plt.ylim(-0.05, 1.05)
    
    # Add horizontal line at perfect accuracy
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1, label='Perfect accuracy')
    
    plt.tight_layout()
    plt.show()
    
    print("\n Plot displayed!")
