from dataclasses import dataclass
from typing import List

import torch
from pharmacophore2mol.data.dag_dataset import Node, _FailedSample, ReplacerCollate
from pharmacophore2mol.data.nodes import (
    SDFLoader,
    FilterElements,
    FilterMaxAtoms,
    RandomJitter,
    ExtractAtomCoords,
    ExtractAtomicNumbers,
    ExtractBondOrderMatrix,
)


@dataclass
class BonderSample:
    coords: torch.Tensor
    atomic_numbers: torch.Tensor
    bond_orders: torch.Tensor
    atom_mask: torch.Tensor
    num_atoms: int


@dataclass
class BonderBatch:
    coords: torch.Tensor
    atomic_numbers: torch.Tensor
    bond_orders: torch.Tensor
    atom_mask: torch.Tensor
    num_atoms: torch.Tensor


class BonderDataset(Node):
    """
    Root dataset node for dense bond-order prediction.

    It composes a small DAG pipeline of reusable data nodes:
    SDFLoader -> FilterElements -> FilterMaxAtoms -> RandomJitter -> ExtractAtomCoords / ExtractAtomicNumbers / ExtractBondOrderMatrix.
    """

    def setup(self, sdf_filepath, allowed_elements=("C", "H", "O", "N"), jitter_sigma=0.05, max_atoms=100):
        source = SDFLoader(sdf_filepath=sdf_filepath)
        filtered = FilterElements(source, allowed_elements=allowed_elements)
        size_filtered = FilterMaxAtoms(filtered, max_atoms=max_atoms)
        jittered = RandomJitter(size_filtered, sigma=jitter_sigma)

        coords = ExtractAtomCoords(jittered)
        atomic_numbers = ExtractAtomicNumbers(jittered)
        bond_orders = ExtractBondOrderMatrix(jittered)

        self.configure_parents([coords, atomic_numbers, bond_orders])
        self.continue_on_error = True

    def forward(self, coords, atomic_numbers, bond_orders):
        if coords is None or atomic_numbers is None or bond_orders is None:
            return None

        coords = torch.as_tensor(coords, dtype=torch.float32)
        atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)
        bond_orders = torch.as_tensor(bond_orders, dtype=torch.long)
        atom_mask = torch.ones(coords.shape[0], dtype=torch.bool)

        return BonderSample(
            coords=coords,
            atomic_numbers=atomic_numbers,
            bond_orders=bond_orders,
            atom_mask=atom_mask,
            num_atoms=coords.shape[0],
        )


#THIS NEEDS TO BE USED IN CONJUNCTION WITH THE USUAL REPLACER COLLATE IF WE WANT TO AVOID VARIABLE LENGTH BATCHES
def bonder_collate_fn(samples: List[BonderSample]) -> BonderBatch:
    samples = [s for s in samples if not isinstance(s, _FailedSample)]
    if len(samples) == 0:
        raise ValueError("All samples in the batch failed. Try using a smaller batch size or cleaner data.")

    batch_size = len(samples)
    max_atoms = max(sample.num_atoms for sample in samples)

    coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    atomic_numbers = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    bond_orders = torch.zeros((batch_size, max_atoms, max_atoms), dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    num_atoms = torch.zeros((batch_size,), dtype=torch.long)

    for batch_idx, sample in enumerate(samples):
        n = sample.num_atoms
        coords[batch_idx, :n] = sample.coords
        atomic_numbers[batch_idx, :n] = sample.atomic_numbers
        bond_orders[batch_idx, :n, :n] = sample.bond_orders
        atom_mask[batch_idx, :n] = sample.atom_mask
        num_atoms[batch_idx] = n

    return BonderBatch(
        coords=coords,
        atomic_numbers=atomic_numbers,
        bond_orders=bond_orders,
        atom_mask=atom_mask,
        num_atoms=num_atoms,
    )


class ReplacerBonderCollate:
    def __init__(self, dataset: BonderDataset):
        self.replacer = ReplacerCollate(dataset, max_retries=5)

    def __call__(self, batch: List[BonderSample]) -> BonderBatch:
        return bonder_collate_fn(self.replacer(batch))


if __name__ == "__main__":
    import os
    import pharmacophore2mol as p2m
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from pharmacophore2mol.data.hub import get_dataset
    
    
    TEST_SDF_PATH = get_dataset(filename="geom_5confs_train.sdf")

    dataset = BonderDataset(
        sdf_filepath=TEST_SDF_PATH,
        allowed_elements=('Si', 'H', 'B', 'C', 'Br', 'Bi', 'Cl', 'F', 'N', 'O', 'S', 'P', 'I'),
        jitter_sigma=0.0,
        max_atoms=100,
    )
    dataset.eval()

    print(f"Testing BonderDataset with {TEST_SDF_PATH.name}")
    print(f"Total samples: {len(dataset)}")

    sample = dataset[0]
    if isinstance(sample, _FailedSample):
        print(f"Sample 0 failed: {sample.error}")
    else:
        print(f"Sample 0 loaded")
        print(f" coords: {sample.coords.shape}")
        print(f" atomic_numbers: {sample.atomic_numbers.shape}")
        print(f" bond_orders: {sample.bond_orders.shape}")
        print(f" atom_mask: {sample.atom_mask.shape}")
        print(f" num_atoms: {sample.num_atoms}")

    print("\nTesting DataLoader collation")
    collate_fn = ReplacerBonderCollate(dataset)
    loader = DataLoader(
        dataset,
        batch_size=min(8, len(dataset)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    for batch in tqdm(loader, desc="Iterating over batches"):
        if batch is None:
            print("Batch collation failed, skipping...")
            continue
        # tqdm.write(f"Batch coords: {batch.coords.shape}")
        # tqdm.write(f"Batch atomic_numbers: {batch.atomic_numbers.shape}")
        # tqdm.write(f"Batch bond_orders: {batch.bond_orders.shape}")
        # tqdm.write(f"Batch atom_mask: {batch.atom_mask.shape}")
        # tqdm.write(f"Batch num_atoms: {batch.num_atoms.shape}")
        # tqdm.write("-" * 50)
