import torch
import numpy as np
import math
import random
from torch.utils.data import Dataset
from rdkit import Chem
from typing import Optional, List


class VoxMolDataset(Dataset):
    """
    VoxMol dataset that reads directly from SDF files.
    Compatible with custom voxelizers.
    
    Parameters
    ----------
    sdf_path : str
        Path to the SDF file containing molecules.
    elements : list
        List of element symbols to include (e.g., ["C", "H", "O", "N", "F"]).
    atomic_radius : float
        Atomic radius to use for all atoms. Default is 0.5Å.
    max_n_atoms : int
        Maximum number of atoms allowed in a molecule. 0 means no filtering.
    apply_augmentation : bool
        Whether to apply rotation and translation augmentation.
    removeHs : bool
        Whether to remove hydrogens when reading SDF. Default False (keep Hs).
    """
    
    def __init__(
        self,
        sdf_path: str,
        elements: List[str] = ["C", "H", "O", "N", "F"],
        atomic_radius: float = 0.5,
        max_n_atoms: int = 80,
        apply_augmentation: bool = True,
        removeHs: bool = False,
    ):
        self.sdf_path = sdf_path
        self.elements = elements
        self.element_to_id = {elem: i for i, elem in enumerate(elements)}
        self.atomic_radius = atomic_radius
        self.max_n_atoms = max_n_atoms
        self.apply_augmentation = apply_augmentation
        
        # Load molecules from SDF
        print(f"Loading molecules from {sdf_path}...")
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=removeHs, sanitize=False)
        
        # Parse and filter molecules
        self.data = []
        for mol in suppl:
            if mol is None:
                continue
                
            # Get coordinates from conformer
            if mol.GetNumConformers() == 0:
                continue
            conf = mol.GetConformer()
            
            # Extract atom data
            coords = []
            atom_types = []
            
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol not in self.element_to_id:
                    continue
                
                pos = conf.GetAtomPosition(atom.GetIdx())
                coords.append([pos.x, pos.y, pos.z])
                atom_types.append(self.element_to_id[symbol])
            
            # Filter by number of atoms
            if len(coords) == 0:
                continue
            if max_n_atoms > 0 and len(coords) >= max_n_atoms:
                continue
            
            self.data.append({
                "coords": np.array(coords, dtype=np.float32),
                "atoms_channel": np.array(atom_types, dtype=np.int64),
            })
        
        print(f"Loaded {len(self.data)} molecules")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        """
        Returns a sample with augmentation applied.
        
        Returns
        -------
        dict with keys:
            - 'coords': np.ndarray of shape (n_atoms, 3)
            - 'atoms_channel': np.ndarray of shape (n_atoms,)
            - 'radius': float, atomic radius
            - 'coords_by_element': dict mapping element names to their coordinates
                                  (for use with custom voxelizer)
        """
        sample_raw = self.data[index]
        
        # Copy data
        coords = sample_raw["coords"].copy()
        atoms_channel = sample_raw["atoms_channel"].copy()
        
        # Apply augmentation
        if self.apply_augmentation:
            coords = self._center_coords(coords)
            coords = self._rotate_coords(coords)
            coords = self._shift_coords(coords, delta=0.5)
        
        # Prepare output
        sample = {
            "coords": coords,
            "atoms_channel": atoms_channel,
            "radius": self.atomic_radius,
        }
        
        # Add format for custom voxelizer (dict by element)
        coords_by_element = {}
        for elem_name, elem_id in self.element_to_id.items():
            mask = atoms_channel == elem_id
            if mask.any():
                coords_by_element[elem_name] = coords[mask]
        
        sample["coords_by_element"] = coords_by_element
        
        return sample
    
    def _center_coords(self, coords: np.ndarray) -> np.ndarray:
        """Center coordinates around center of mass."""
        center = np.mean(coords, axis=0)
        return coords - center
    
    def _shift_coords(self, coords: np.ndarray, delta: float = 0.5) -> np.ndarray:
        """Add random translation noise."""
        noise = (np.random.rand(1, 3) - 0.5) * 2 * delta
        return coords + noise
    
    def _rotate_coords(self, coords: np.ndarray) -> np.ndarray:
        """Apply random rotation."""
        rot_matrix = self._random_rot_matrix()
        # Center first
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        # Rotate
        coords_rotated = coords_centered @ rot_matrix.T
        return coords_rotated
    
    def _random_rot_matrix(self) -> np.ndarray:
        """Generate random 3D rotation matrix."""
        theta_x = random.uniform(0, 2) * math.pi
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x)],
            [0, math.sin(theta_x), math.cos(theta_x)],
        ])
        
        theta_y = random.uniform(0, 2) * math.pi
        rot_y = np.array([
            [math.cos(theta_y), 0, -math.sin(theta_y)],
            [0, 1, 0],
            [math.sin(theta_y), 0, math.cos(theta_y)],
        ])
        
        theta_z = random.uniform(0, 2) * math.pi
        rot_z = np.array([
            [math.cos(theta_z), -math.sin(theta_z), 0],
            [math.sin(theta_z), math.cos(theta_z), 0],
            [0, 0, 1],
        ])
        
        return rot_z @ rot_y @ rot_x


def collate_fn(batch):
    """
    Custom collate function for VoxMolDataset.
    Since molecules have different numbers of atoms, we can't stack directly.
    Returns a list of samples instead.
    """
    return batch


# Example usage
if __name__ == "__main__":
    from pharmacophore2mol.data.voxelizer import Voxelizer
    import matplotlib.pyplot as plt
    
    # Create dataset
    dataset = VoxMolDataset(
        sdf_path="path/to/your/molecules.sdf",
        elements=["C", "H", "O", "N", "F"],
        atomic_radius=0.5,
        max_n_atoms=80,
        apply_augmentation=True,
    )
    
    # Create voxelizer (matching VoxMol's setup)
    voxelizer = Voxelizer(
        channels=["C", "H", "O", "N", "F"],
        resolution=0.25,  # VoxMol uses 0.25Å resolution
        mode="gaussian",
        std=0.93 * 0.5,  # VoxMol's 0.93 * atomic_radius formula
        pooling="max"  # or "pyuul" if you implement that mode
    )
    
    # Get a sample
    sample = dataset[0]
    print(f"Coords shape: {sample['coords'].shape}")
    print(f"Atom types: {sample['atoms_channel']}")
    
    # Voxelize using your voxelizer
    voxel_grid = voxelizer.voxelize(
        sample["coords_by_element"],
        min_grid_shape=(32, 32, 32),  # QM9 uses 32³, GEOM-drugs uses 64³
        allow_negative_coords=True
    )
    
    print(f"Voxel grid shape: {voxel_grid.shape}")
    
    # Visualize a slice
    plt.imshow(voxel_grid[0, 16, :, :])  # C channel, middle slice
    plt.title("Carbon channel")
    plt.colorbar()
    plt.show()
    
    # DataLoader example
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,  # Use custom collate since molecules have different sizes
        num_workers=2
    )
    
    for batch in dataloader:
        print(f"Batch size: {len(batch)}")
        # Voxelize each molecule in the batch
        for sample in batch:
            voxels = voxelizer.voxelize(
                sample["coords_by_element"],
                min_grid_shape=(32, 32, 32),
                allow_negative_coords=True
            )
            print(f"Voxelized to shape: {voxels.shape}")
        break