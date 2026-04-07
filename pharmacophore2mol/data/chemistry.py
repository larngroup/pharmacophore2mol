from rdkit import Chem


ALLOWED_VALENCES = {
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
    'Se': [2, 4, 6],
}


def is_valence_stable(symbol: str, valency: float, charge: int = 0) -> bool:
    """
    Determine whether an atomic valency is chemically stable for a given element and formal charge.
    """
    if symbol not in ALLOWED_VALENCES:
        return False

    allowed = ALLOWED_VALENCES[symbol]
    if isinstance(allowed, int):
        return valency == float(allowed)
    if isinstance(allowed, list):
        return valency in [float(x) for x in allowed]
    if isinstance(allowed, dict):
        expected = allowed.get(charge, None)
        if expected is None:
            expected = allowed.get(0, None)
        if expected is None:
            return False
        if isinstance(expected, int):
            return valency == float(expected)
        return valency in [float(x) for x in expected]

    return False


def get_atom_valence(atom) -> float:
    """
    Compute the valency of a single RDKit atom by summing bond orders.
    """
    valency = 0.0
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
    return valency


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

    n_stable_atoms = 0
    n_total_atoms = mol.GetNumAtoms()

    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        valency = get_atom_valence(atom)

        if is_valence_stable(atom_symbol, valency, charge):
            n_stable_atoms += 1

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
