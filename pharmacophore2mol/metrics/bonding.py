import torch
from rdkit import Chem
from pharmacophore2mol.data.chemistry import is_valence_stable

BOND_ORDER_WEIGHTS = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5], dtype=torch.float32)


def valid_valence_rate(
    logits: torch.Tensor,
    atomic_numbers: torch.Tensor,
    atom_mask: torch.Tensor,
) -> tuple[float, int]:
    """Calculate the valid valence rate for a batch of predicted bond orders.
    
    Parameters
    ----------
        logits: [B, N, N, num_classes] - raw output from the model before softmax
        atomic_numbers: [B, N] - atomic numbers of the atoms in the batch
        atom_mask: [B, N] - boolean mask indicating valid atoms (1 for valid, 0 for padding)
    Returns
    -------
        A tuple of (valid_valence_rate, total_atoms_evaluated)
        
    """
    predictions = torch.argmax(logits, dim=-1)
    bond_weights = BOND_ORDER_WEIGHTS.to(logits.device)[predictions]
    bond_sum = bond_weights.sum(dim=-1)

    valid_atoms = 0
    total_atoms = 0

    periodic_table = Chem.GetPeriodicTable()
    batch_size, num_nodes = atomic_numbers.shape

    for batch_idx in range(batch_size):
        for atom_idx in range(num_nodes):
            if not atom_mask[batch_idx, atom_idx].item():
                continue

            z = int(atomic_numbers[batch_idx, atom_idx].item())
            if z == 0:
                continue

            symbol = periodic_table.GetElementSymbol(z)
            valency = float(bond_sum[batch_idx, atom_idx].item())
            if is_valence_stable(symbol, valency):
                valid_atoms += 1
            total_atoms += 1

    if total_atoms == 0:
        return 0.0, 0

    return valid_atoms / total_atoms, total_atoms
