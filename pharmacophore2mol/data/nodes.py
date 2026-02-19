import numpy as np
from pharmacophore2mol.data.dag_dataset import Node


from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm
from collections.abc import Iterable
from pharmacophore2mol.data.utils import CustomSDMolSupplier, get_atom_coords, set_atom_coords, mol_to_atom_dict
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

class FilterElements(Node):
    """
    Node for filtering molecules that contain elements not in the allowed list.
    Returns None if the molecule contains any element not in the whitelist.
    
    Parameters
    ----------
    allowed_elements : list of str
        List of allowed atomic symbols (e.g. ['C', 'H', 'O', 'N']).
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

class ExtractCoords(Node):
    """
    Node for extracting coordinates from an RDKit molecule or a Pharmacophore object.
    Returns a dictionary of element symbols (or feature families) to coordinate arrays (N, 3).
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
    Returns a Pharmacophore object.

    Parameters
    ----------
    ignore_directions : bool
        Whether to skip direction vector calculation (faster if only points are needed). Default is False.

    Input
    -----
    mol : rdkit.Chem.Mol
        Molecule with 3D conformer.
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