import numpy as np
from pharmacophore2mol.data.dag_dataset import Node


from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm
from collections.abc import Iterable
from pharmacophore2mol.data.utils import CustomSDMolSupplier
from pharmacophore2mol.data.voxelizer import Voxelizer, get_frag_count, fragment_voxel_grid
from pharmacophore2mol.data.pharmacophore import Pharmacophore, PHARMACOPHORE_CHANNELS




class SDFLoader(Node):
    """
    Base loader node for loading molecules from an SDF file.
    
    Parameters
    ----------
    sdf_filepath : str | Path
        Path to the SDF file.
    """
    def setup(self, sdf_filepath):
        self.sdf_filepath = sdf_filepath
        self.mol_supplier = CustomSDMolSupplier(self.sdf_filepath)

    def __len__(self):
        return len(self.mol_supplier)
    
    def forward(self, index):
        return self.mol_supplier[index] #if there's an error then this returns None, which is filtered out anyways

class ExtractAtomCoords(Node):
    """
    Node for extracting atom coordinates from an RDKit molecule.
    """
    def forward(self, mol):
        if mol is None:
            return None
        conf = mol.GetConformer()
        coords = conf.GetPositions()  # shape (n_atoms, 3)
        return coords

class RandomRotate(Node):
    """
    Randomly rotate either an RDKit Mol (3D conformer) or a NumPy array of shape (N,3).
    """
    def setup(self, max_angles=(359, 359, 359), center="mean"):
        # max_angles in degrees (per-axis). Clip to [0,359] and store in radians.
        self.max_angles = np.deg2rad(np.clip(np.array(max_angles, dtype=float), 0, 359))
        # center can be "mean" or a length-3 iterable of coordinates
        self.center = center

    def _rotation_matrix(self, angles):
        x, y, z = angles
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(x), -np.sin(x)],
                        [0, np.sin(x),  np.cos(x)]])
        Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                        [0, 1, 0],
                        [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0],
                        [np.sin(z),  np.cos(z), 0],
                        [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _rotate_coords(self, coords, rng):
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be shape (N,3), got {coords.shape}")
        # choose center
        if self.center == "mean":
            c = coords.mean(axis=0)
        else:
            c = np.asarray(self.center, dtype=float)
            if c.shape != (3,):
                raise ValueError("center must be 'mean' or an iterable of length 3")
        # sample angles in [0, max_angle] per axis using provided RNG
        angles = rng.uniform(0.0, self.max_angles)
        R = self._rotation_matrix(angles)
        return (coords - c) @ R.T + c

    def forward(self, data, seed):
        """
        Parameters
        ----------
        data : rdkit.Chem.Mol or np.ndarray
        """
        rng = np.random.default_rng(seed)

        # RDKit molecule case
        try:
            is_mol = isinstance(data, Chem.Mol)
        except Exception:
            is_mol = False

        if is_mol:
            if data.GetNumConformers() == 0:
                raise ValueError("RDKit Mol has no conformer to rotate.")
            # extract coordinates
            conf = data.GetConformer(0)
            coords = conf.GetPositions()
            new_coords = self._rotate_coords(coords, rng)
            conf.SetPositions(new_coords)
            return data

        # numpy array case
        if isinstance(data, np.ndarray):
            return self._rotate_coords(data, rng)

        # unknown type: pass through unchanged (or raise)
        raise TypeError("RandomRotate only supports rdkit.Chem.Mol or numpy.ndarray (N,3).")

