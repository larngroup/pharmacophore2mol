import numpy as np
from pharmacophore2mol.data.dag_dataset import Node


from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm
from collections.abc import Iterable
from pharmacophore2mol.data.utils import CustomSDMolSupplier, get_atom_coords, set_atom_coords, mol_to_atom_dict
from pharmacophore2mol.data.voxelizer import Voxelizer, get_frag_count, fragment_voxel_grid
from pharmacophore2mol.data.pharmacophore import Pharmacophore, PHARMACOPHORE_CHANNELS


__all__ = [ #not working for vscode for some reason
    "SDFLoader",
    "FilterElements",
    "FilterMaxAtoms",
    "ExtractCoords",
    "ExtractAtomCoords",
    "ExtractAtomicNumbers",
    "ExtractBondOrderMatrix",
    "ExtractPharmacophore",
    "RandomRotate",
    "RandomFlip",
    "RandomJitter",
    "BoxSelector",
    "Voxelize"
]

class SDFLoader(Node):
    """
    Base loader node for loading molecules from an SDF file.
    
    This loader is worker-safe: the underlying RDKit supplier is recreated
    lazily in each process, so PyTorch DataLoader can use multiple workers.
    
    Parameters
    ----------
    sdf_filepath : str | Path
        Path to the SDF file.

    Returns
    -------
    rdkit.Chem.Mol or None
        The loaded molecule, or None if parsing fails.
    """
    def setup(self, sdf_filepath):
        self.sdf_filepath = sdf_filepath
        self.mol_supplier = None

    def _ensure_supplier(self):
        if self.mol_supplier is None:
            self.mol_supplier = CustomSDMolSupplier(self.sdf_filepath)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('mol_supplier', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mol_supplier = None

    def __len__(self):
        self._ensure_supplier()
        return len(self.mol_supplier)
    
    def forward(self, index):
        self._ensure_supplier()
        return self.mol_supplier[index] #if there's an error then this returns None, which is filtered out anyways

class FilterElements(Node):
    """
    Node for filtering molecules that contain elements not in the allowed list.
    Returns None if the molecule contains any element not in the whitelist.
    
    Parameters
    ----------
    allowed_elements : list of str
        List of allowed atomic symbols (e.g. ['C', 'H', 'O', 'N']).

    Input
    -----
    mol : rdkit.Chem.Mol or None
        The molecule to filter.

    Returns
    -------
    rdkit.Chem.Mol or None
        The original molecule if it passes the filter, otherwise None.
    """
    def setup(self, allowed_elements):
        self.allowed_elements = set(allowed_elements)

    def forward(self, mol):
        if mol is None:
            return None
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in self.allowed_elements:
                return None
        return mol

class FilterMaxAtoms(Node):
    """
    Filters molecules larger than a maximum atom count.

    Parameters
    ----------
    max_atoms : int
        Maximum allowed atom count. Molecules with more atoms are rejected.

    Input
    -----
    mol : rdkit.Chem.Mol or None
        The molecule to evaluate.

    Returns
    -------
    rdkit.Chem.Mol or None
        The original molecule if it has <= max_atoms, otherwise None.
    """
    def setup(self, max_atoms):
        self.max_atoms = int(max_atoms)

    def forward(self, mol):
        if mol is None:
            return None
        if mol.GetNumAtoms() > self.max_atoms:
            return None
        return mol

class ExtractCoords(Node):
    """
    Node for extracting coordinates from an RDKit molecule or a Pharmacophore object.
    
    Input
    -----
    data : rdkit.Chem.Mol or Pharmacophore or None
        The object to extract coordinates from.

    Returns
    -------
    dict or None
        A dictionary of element symbols (or feature families) to coordinate arrays (N, 3).
    """
    def forward(self, data):
        if data is None:
            return None
        
        if isinstance(data, Chem.Mol):
            return mol_to_atom_dict(data)
        
        if isinstance(data, Pharmacophore):
            return data.to_dict(np_format=True)

        raise TypeError(f"ExtractCoords only supports rdkit.Chem.Mol or Pharmacophore objects. Got {type(data)}.")

class ExtractPharmacophore(Node):
    """
    Node for extracting pharmacophore features from an RDKit molecule.

    Parameters
    ----------
    ignore_directions : bool
        Whether to skip direction vector calculation (faster if only points are needed). Default is False.

    Input
    -----
    mol : rdkit.Chem.Mol or None
        Molecule with 3D conformer.

    Returns
    -------
    Pharmacophore or None
        The extracted Pharmacophore object.
    """
    def setup(self, ignore_directions=False):
        self.ignore_directions = ignore_directions

    def forward(self, mol):
        if mol is None: 
            return None
        
        return Pharmacophore(mol, ignore_directions=self.ignore_directions)

class RandomRotate(Node):
    """
    Randomly rotate coordinates using an Azimuth/Tilt control model.
    Accepts RDKit Mol or NumPy array (N,3).

    This method generates a uniform rotation defined by a primary axis (Azimuth/Pole)
    and a deviation (Tilt) from that pole. This creates a perfect circular cone of 
    probability density around the chosen axis, avoiding the pole-bunching of naive Euler.

    Parameters
    ----------
    azimuth_axis : int {0, 1, 2} or str {'x', 'y', 'z'}
        The primary axis to rotate around (spin). Default is 2 (Z).
    max_spin : float
        Maximum rotation angle around the azimuth axis in degrees. Default is 180 (free spin).
        This defines the range [-max_spin, max_spin]. So 180 corresponds to a full 360 spin.
    max_tilt : float
        Maximum tilt angle (deviation) from the azimuth axis in degrees. Default is 180. 
        If 0, it is a pure spin around the azimuth axis. If 180, it covers the full sphere uniformly.

    Input
    -----
    data : rdkit.Chem.Mol or np.ndarray
        If rdkit.Chem.Mol, it must have at least one conformer.

    Returns
    -------
    rdkit.Chem.Mol or np.ndarray
        The rotated data, matching the input type.
    """
    def setup(self, azimuth_axis=2, max_spin=180, max_tilt=180):
        if isinstance(azimuth_axis, str):
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            self.azimuth_axis = axis_map[azimuth_axis.lower()]
        else:
            self.azimuth_axis = int(azimuth_axis)

        self.max_spin = np.deg2rad(float(max_spin))
        self.max_tilt = np.deg2rad(float(max_tilt))
        

        if self.max_spin > np.pi:
            self.max_spin = np.pi

    def _get_axis_tilt_rotation(self, rng):
        azimuth = rng.uniform(-self.max_spin, self.max_spin)
        
        # need to sample uniformly but in cosine space, see Arvo algorithm, else there's bunching near the poles
        z_min = np.cos(self.max_tilt)
        u = rng.uniform(z_min, 1.0)
        tilt = np.arccos(u)
        
        tilt_direction = rng.uniform(0, 2 * np.pi) #tilt direction doees not depend on azimuth direction

        # need a perpendicular axis to tip AROUND, if azimuth is z, then it can be x or y, we just pick one
        perp_axis = (self.azimuth_axis - 1) % 3 
        phi = tilt_direction
        
        R_align = self._R_axis(self.azimuth_axis, -phi) #spins to align the tilt axis with the selected perp_axis
        R_tilt = self._R_axis(perp_axis, tilt) #tilts around the perp_axis
        R_final = self._R_axis(self.azimuth_axis, azimuth + phi) #spins back to original azimuth and adds the random spin
        
        return R_final @ R_tilt @ R_align #align to azimuth axis, tilt, then undo align and spin
    
    def _R_axis(self, axis_idx, angle):
        c, s = np.cos(angle), np.sin(angle)
        if axis_idx == 0:
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis_idx == 1:
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis_idx == 2:
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _rotate_coords(self, coords, rng):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be shape (N,3), got {coords.shape}")

        R = self._get_axis_tilt_rotation(rng)
        
        center = coords.mean(axis=0)
        return (coords - center) @ R.T + center

    def forward(self, data, seed):
        """
        Parameters
        ----------
        data : rdkit.Chem.Mol or np.ndarray
        """
        rng = np.random.default_rng(seed)

        if isinstance(data, Chem.Mol):
            coords = get_atom_coords(data)
            new_coords = self._rotate_coords(coords, rng)
            set_atom_coords(data, new_coords)
            return data

        if isinstance(data, np.ndarray):
            return self._rotate_coords(data, rng)
            
        raise TypeError("RandomRotate only supports rdkit.Chem.Mol or numpy.ndarray (N,3).")

class RandomFlip(Node):
    """
    Randomly flip coordinates along the centroid (mean center, NOT the bounding circle center).
    Accepts either an RDKit Mol (3D conformer) or a NumPy array of shape (N,3). 
    
    This transformation mirrors the molecule relative to its own centroid along the specified axes.
    Allowed planes of reflection are defined by the `axes` parameter. For example, if `axes=(0, 2)`, the molecule can be flipped along the X and Z axes, but not Y.
    
    Parameters
    ----------
    axes : tuple of int
        Which axes are allowed to be flipped (meaning it can flip over the planes normal to the allowed axes that intersect the centroid). Default is (0, 1, 2) which allows X, Y, or Z flips.
    p : float
        Probability of applying a flip. Default is 0.5.

    Input
    -----
    data : rdkit.Chem.Mol or np.ndarray
        The data to flip.

    Returns
    -------
    rdkit.Chem.Mol or np.ndarray
        The flipped data, matching the input type.
    """

    def setup(self, axes=(0, 1, 2), p=0.5):
        self.axes = list(axes)
        self.p = p

    def _flip_coords(self, coords, rng):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be shape (N,3), got {coords.shape}")

        if rng.random() > self.p:
            return coords
            
        # select just one axis to flip, otherwise we get the same molecule back with 2 flips canceling out
        axis_to_flip = rng.choice(self.axes)

        # using mean centroid
        center = coords.mean(axis=0)

        coords[:, axis_to_flip] = -coords[:, axis_to_flip] + 2 * center[axis_to_flip]
            
        return coords

    def forward(self, data, seed):
        rng = np.random.default_rng(seed)

        if isinstance(data, Chem.Mol):
            coords = get_atom_coords(data)
            self._flip_coords(coords, rng)
            set_atom_coords(data, coords)
            return data

        if isinstance(data, np.ndarray):
            return self._flip_coords(data, rng)

        raise TypeError("RandomFlip only supports rdkit.Chem.Mol or numpy.ndarray (N,3).")


class RandomJitter(Node):
    """
    Adds Gaussian jitter to atomic coordinates in an RDKit molecule.

    This node behaves like other Mol transform nodes in the DAG pipeline.
    Noise is only injected when the node is in training mode.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian coordinate jitter in angstroms.

    Input
    -----
    data : rdkit.Chem.Mol or None
        Molecule to perturb.

    Returns
    -------
    rdkit.Chem.Mol or None
        The same molecule object with jittered coordinates in training mode,
        otherwise the original molecule.
    """
    def setup(self, sigma=0.05):
        self.sigma = float(sigma)

    def forward(self, data, seed):
        if data is None:
            return None

        if not self.training or self.sigma <= 0.0:
            return data

        if seed is None:
            seed = np.random.SeedSequence().entropy
        rng = np.random.default_rng(int(seed))

        coords = get_atom_coords(data)
        noise = rng.normal(0.0, self.sigma, size=coords.shape).astype(np.float32)
        set_atom_coords(data, coords + noise)
        return data


class ExtractAtomCoords(Node):
    """
    Extracts raw atom coordinates from an RDKit molecule.

    Input
    -----
    mol : rdkit.Chem.Mol or None
        Molecule from which to extract coordinates.

    Returns
    -------
    np.ndarray or None
        Array of shape (N, 3) with atom coordinates.
    """
    def forward(self, mol):
        if mol is None:
            return None
        return get_atom_coords(mol)


class ExtractAtomicNumbers(Node):
    """
    Extracts atomic numbers from an RDKit molecule.

    Input
    -----
    mol : rdkit.Chem.Mol or None
        Molecule from which to extract atomic numbers.

    Returns
    -------
    np.ndarray or None
        1D integer array containing atomic numbers.
    """
    def forward(self, mol):
        if mol is None:
            return None
        return np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)


class ExtractBondOrderMatrix(Node):
    """
    Converts an RDKit molecule into a symmetric bond order matrix.

    Input
    -----
    mol : rdkit.Chem.Mol or None
        Molecule from which to extract bond orders.

    Returns
    -------
    np.ndarray or None
        Pairwise bond order matrix of shape (N, N) with classes:
        0=no bond, 1=single, 2=double, 3=triple, 4=aromatic.
    """
    def forward(self, mol):
        if mol is None:
            return None

        num_atoms = mol.GetNumAtoms()
        bond_orders = np.zeros((num_atoms, num_atoms), dtype=np.int64)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                label = 1
            elif bond_type == Chem.BondType.DOUBLE:
                label = 2
            elif bond_type == Chem.BondType.TRIPLE:
                label = 3
            elif bond_type == Chem.BondType.AROMATIC:
                label = 4
            else:
                label = 0

            bond_orders[i, j] = label
            bond_orders[j, i] = label

        return bond_orders


class Voxelize(Node):
    """
    Voxelizes a point cloud into a 3D grid.

    It expects two inputs from its parents, the points dictionary and the bounding box for the voxel grid.

    Parameters
    ----------
    channels : list of str
        List of channel names to include in the voxel grid.
    resolution : float, optional
        Resolution of the voxel grid in Angstroms (size of one voxel edge). Default is 0.25.
    mode : str, optional
        Voxelization mode (e.g., 'gaussian', 'binary'). Default is 'gaussian'.
    force_shape : tuple of int, optional
        If provided, forces the output grid shape to this specific size.
    **voxelizer_kwargs : dict
        Additional arguments passed to the Voxelizer constructor (e.g. `std`, `pooling`).

    Input
    -----
    points : dict
        A dictionary mapping channel names to numpy arrays of coordinates (N, 3).
    box : tuple
        A tuple containing `(center, side_length)`, where `center` is a 
        (x, y, z) coordinate and `side_length` is a float.

    Returns
    -------
    np.ndarray or None
        A 4D numpy array of shape (C, D, H, W) containing the voxelized data, 
        or None if inputs are missing.
    """
    def setup(self, channels: list[str], resolution: float = 0.25, mode: str = "gaussian", force_shape: tuple = None, **voxelizer_kwargs):
        self.voxelizer = Voxelizer(channels=channels, resolution=resolution, mode=mode, **voxelizer_kwargs)
        self.force_shape = force_shape

    def change_channels(self, new_channels: list[str]):
        """
        Update the channels used by the underlying voxelizer.
        """
        self.voxelizer.channels = new_channels

    def forward(self, points, box):
        if points is None or box is None:
            return None
            
        center, side_length = box
        return self.voxelizer.voxelize(
            points=points, 
            center=center, 
            side_length=side_length, 
            force_shape=self.force_shape
        )

class BoxSelector(Node):
    """
    Selects a cubic bounding box (center, side_length) from a point cloud.

    The admissible region is defined by the bounding box of the input coordinates, 
    expanded or shrunk by a `padding` value. The box center is then chosen based 
    on the specified `mode`.

    Parameters
    ----------
    side_length : float
        The side length of the cubic box to select.
    mode : str, optional
        The mode for selecting the box center. Options are:
        - "center": The center of the admissible region (bounding box of points).
        - "random": A random point uniformly sampled within the admissible region.
        - "random_feature": A random coordinate chosen directly from the input points.
        Default is "center".
    padding : float, optional
        Padding added to the bounding box of the input points to define the 
        admissible region. Can be negative. Default is 0.0.

    Input
    -----
    points : dict or np.ndarray
        The input point cloud. Can be a dictionary mapping channel names to 
        coordinate arrays (N, 3), or a single numpy array of shape (N, 3).

    Returns
    -------
    tuple or None
        A tuple `(center, side_length)` where `center` is a numpy array of shape (3,) 
        and `side_length` is a float. Returns None if the input is empty or None.
    """
    def setup(self, side_length: float, mode: str = "center", padding: float = 0.0):
        valid_modes = ["center", "random", "random_feature"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")
        
        self.side_length = float(side_length)
        self.mode = mode
        self.padding = float(padding)

    def forward(self, points, seed):
        if points is None:
            return None

        if isinstance(points, dict):
            arrays = [arr for arr in points.values() if len(arr) > 0]
            if not arrays:
                return None
            all_coords = np.vstack(arrays)
        elif isinstance(points, np.ndarray):
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Expected points array of shape (N, 3), got {points.shape}")
            if len(points) == 0:
                return None
            all_coords = points
        else:
            raise TypeError(f"Expected points to be dict or np.ndarray, got {type(points)}")

        rng = np.random.default_rng(seed)

        if self.mode == "random_feature":
            idx = rng.integers(0, len(all_coords))
            center = all_coords[idx]
            return (center, self.side_length)

        min_coords = np.min(all_coords, axis=0) - self.padding
        max_coords = np.max(all_coords, axis=0) + self.padding

        if self.mode == "center":
            center = (min_coords + max_coords) / 2.0
            return (center, self.side_length)

        elif self.mode == "random":
            # sample uniformly within the padded bounding box
            center = rng.uniform(low=min_coords, high=max_coords)
            return (center, self.side_length)
        

# __all__ = [
#     name for name, obj in globals().items()
#     if getattr(obj, '__module__', None) == __name__ and not name.startswith('_')
# ]