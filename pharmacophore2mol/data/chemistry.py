from rdkit import Chem




def get_mol_stability(mol) -> float:
    """
    Computes the stability of a molecule based on atom valencies and charges.
    
    A stable atom is defined as one whose valency matches expected values
    for its element type and formal charge.
    
    Args:
        mol: RDKit Mol object
    Returns:
        Stability score: fraction of stable atoms (0.0 to 1.0)
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    
    # dictionary of allowed valencies per atom type and charge
    # from MiDi/VoxMol implementation
    allowed_bonds = {
        'H': {0: 1, 1: 0, -1: 0},
        'C': {0: [3, 4], 1: 3, -1: 3},
        'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},
        'O': {0: 2, 1: 3, -1: 1},
        'F': {0: 1, -1: 0},
        'B': 3,
        'Al': 3,
        'Si': 4,
        'P': {0: [3, 5], 1: 4},
        'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
        'Cl': 1,
        'As': 3,
        'Br': {0: 1, 1: 2},
        'I': 1,
        'Hg': [1, 2],
        'Bi': [3, 5],
        'Se': [2, 4, 6]
    }
    
    n_stable_atoms = 0
    n_total_atoms = mol.GetNumAtoms()
    
    for atom in mol.GetAtoms():
        # Get atom properties
        atom_symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        
        # Calculate valency (sum of bond orders)
        valency = 0
        for bond in atom.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                valency += 1
            elif bond_type == Chem.BondType.DOUBLE:
                valency += 2
            elif bond_type == Chem.BondType.TRIPLE:
                valency += 3
            elif bond_type == Chem.BondType.AROMATIC:
                valency += 1.5
            # ignoring other bond types (like DATIVE)
        
        if atom_symbol not in allowed_bonds:
            # Unknown atom type, assume unstable
            continue
        
        possible_bonds = allowed_bonds[atom_symbol]
        
        is_stable = False
        
        if isinstance(possible_bonds, int):
            is_stable = (valency == possible_bonds)
            
        elif isinstance(possible_bonds, list):
            is_stable = (valency in possible_bonds)
            
        elif isinstance(possible_bonds, dict):
            # charge-dependent valencies (e.g., H, C, N, O)
            if charge in possible_bonds:
                expected_bonds = possible_bonds[charge]
            else:
                # use neutral as default
                expected_bonds = possible_bonds.get(0, None)
            
            if expected_bonds is None:
                is_stable = False
            elif isinstance(expected_bonds, int):
                is_stable = (valency == expected_bonds)
            elif isinstance(expected_bonds, list):
                is_stable = (valency in expected_bonds)
        
        if is_stable:
            n_stable_atoms += 1
        # else:
        #     print(f"Unstable atom: {atom_symbol} (charge: {charge}, valency: {valency})")
    
    return n_stable_atoms / n_total_atoms if n_total_atoms > 0 else 0.0


def rdkit_is_valid(mol) -> bool:
    """
    Checks if an RDKit Mol object is valid (not None and has atoms), according to rdkit criteria.
    Chem.SanitizeMol should be avoided as it has strange behaviours with hydrogens and valencies.
    """
    if mol is not None and mol.GetNumAtoms() > 0:
        return True
    return False

def get_number_of_components(mol) -> int:
    """
    Returns the number of disconnected components in a molecule.
    
    Args:
        mol: RDKit Mol object
    Returns:
        Number of components (int)
    """
    if mol is None:
        return 0
    frags = Chem.GetMolFrags(mol, asMols=False)
    return len(frags)