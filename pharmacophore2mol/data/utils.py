import os
import logging
import sys
import contextlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from rdkit import Chem
from openbabel import pybel
from glob import glob
import pharmacophore2mol as p2m

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_openbabel_warnings():
    """
    Context manager to suppress OpenBabel stderr output.
    Only suppresses if logging level is INFO or higher (not DEBUG).
    """
    current_log_level = logging.getLogger().getEffectiveLevel()
    
    if current_log_level > logging.DEBUG:
        # Suppress OpenBabel C++ stderr output by redirecting file descriptor
        import tempfile
        
        # Save the original stderr file descriptor
        stderr_fd = sys.stderr.fileno()
        old_stderr_fd = os.dup(stderr_fd)
        
        # Redirect stderr to devnull
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
        
        try:
            yield
        finally:
            # Restore original stderr
            os.dup2(old_stderr_fd, stderr_fd)
            os.close(old_stderr_fd)
    else:
        # In verbose/debug mode, don't suppress anything
        yield

# def translate_points_to_positive(points: np.ndarray) -> np.ndarray:
#     """
#     Translate a point cloud to the all positive octant, so that all coordinates are non-negative.
#     """
#     # Find the minimum coordinates in each dimension
#     min_coords = np.min(points, axis=0)

#     # Calculate the translation vector to shift all points to positive coordinates
#     translation_vector = -min_coords

#     # Translate the points
#     translated_points = points + translation_vector

#     return translated_points


def get_mol_supplier(mols_filepath, remove_hs=False, sanitize=True, strict_parsing=False):
    """Just a wrapper to RDKit's SDMolSupplier with my own defaults, for clarity.
    Snake case cuz we pythonic

    Parameters
    ----------
    mols_filepath : str
        Path to the SDF file containing the molecules.
    remove_hs : bool, optional
        Whether to remove hydrogens from the molecules. Default is False.
    sanitize : bool, optional
        Whether to sanitize the molecules. Default is True.
    strict_parsing : bool, optional
        Whether to use strict parsing. Default is False.
    Returns
    -------
    Chem.SDMolSupplier
        RDKit SDMolSupplier object to iterate over the molecules in the SDF file.
    """
    return Chem.SDMolSupplier(mols_filepath, removeHs=remove_hs, sanitize=sanitize, strictParsing=strict_parsing)


#maryam the data agugmentation goes here

def translate_mol(mol: Chem.Mol, translation_vector: np.ndarray) -> Chem.Mol:
    """
    Translate a molecule according to the translation vector.
    Only works for 3D conformers.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("The molecule has no conformers. Please add a conformer before translating.")

    # Get the coordinates of the first conformer
    conf = mol.GetConformer(0)
    coords = conf.GetPositions()

    # Translate the coordinates
    new_coords = coords + translation_vector

    # Set the new coordinates to the conformer
    conf.SetPositions(new_coords)

    return mol


def get_translation_vector(points: np.ndarray, padding=0) -> np.ndarray:
    """
    Get the translation vector to translate the points to the origin.
    This is useful to translate the points back to their original coordinates after voxelization.

    Parameters
    ----------
    points : np.ndarray
        The points to be translated. Should be a 2D array with shape (#points, 3).
    padding : int, optional
        The padding to be added to the translation vector, so that the closest point to the edge of the positive octant (x>0, y>0, z>0) is <padding> Angstroms. Default is 0.
    """
    if points.shape[1] != 3 or len(points.shape) != 2:
        raise ValueError(f"Points should be a 2d array with shape (#points, 3), but got {points.shape}")
    return -np.min(points, axis=0) + padding


def mol_to_atom_dict(mol: Chem.Mol) -> dict:
    """
    Convert a molecule to a dictionary of atom coordinates.

    The dictionary has the element symbols as keys and the coordinates of the atoms as values.

    Example::

        {
            'C': np.array([
                [1.27, 2.36, 3.45],
                [1.0, 5.45, 3.358]
            ]),
            'H': np.array([
                [2.12, 1.79, 5.465]
            ]),
            ...
        }
    """
    atom_coords = mol.GetConformer().GetPositions()
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_types = np.array(atom_types)
    atom_coords = np.array(atom_coords)
    atom_dict = {}
    for atom_type in list(np.unique(atom_types)):
        atom_dict[str(atom_type)] = atom_coords[atom_types == atom_type]

    return atom_dict


def plot_voxel_grid_sweep(voxel_grid: np.ndarray, title: str = "Voxel Grid Sweep"):
    """
    Visualize a 3D voxel grid by sweeping through the z-axis and plotting each slice.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")


    def update(z):
        # Update the image for the current frame
        slice_ = voxel_grid[:, :, z]
        plt.cla()
        ax.set_title(title + f" (z={z})")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        img = ax.imshow(slice_, cmap='viridis', animated=True, vmin=0, vmax=1)
        return img

    # Create an animation
    ani = FuncAnimation(fig, update, frames=range(voxel_grid.shape[2]), interval=200)

    plt.show()

    return ani


def sweep_voxel_grid(voxel_grid: np.ndarray):
    ...


class RandomRotateMolTransform:
    """
    Randomly rotate a molecule around its center of mass.
    """
    def __init__(self, angles: tuple=(359, 359, 359)):
        """
        Initialize the transform with the given angles.
        For a full rotation, with uniform probability, set max angle to 359 degrees (Default). Higher values will be clipped to 359.
        WARNING: the center of mass is not the same as the center of the bounding box.
        This may cause the molecule to be translated a bit. An additional alignment translation may compensate for this.

        Parameters
        ----------
        angles : tuple
            The maximum angles (degrees) to rotate around the x, y and z axes.
        """
        angles = np.clip(angles, 0, 359)
        angles = np.deg2rad(angles)  # Convert angles to radians
        self.x, self.y, self.z = angles

    def _get_rotation_matrix(self, angles: tuple) -> np.ndarray:
        """
        Get the rotation matrix for the given axis angles.

        Parameters
        ----------
        angles : tuple
            The angles (radians) to rotate around the x, y and z axes.
        """
        x, y, z = angles
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(x), -np.sin(x)],
                       [0, np.sin(x), np.cos(x)]])

        Ry = np.array([[np.cos(y), 0, np.sin(y)],
                       [0, 1, 0],
                       [-np.sin(y), 0, np.cos(y)]])

        Rz = np.array([[np.cos(z), -np.sin(z), 0],
                       [np.sin(z), np.cos(z), 0],
                       [0, 0, 1]])

        return Rz @ Ry @ Rx
        

    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        # Get the geometric center of the molecule
        conf = mol.GetConformer(0)
        coords = conf.GetPositions()
        center = np.mean(coords, axis=0) #we can use the center of mass, but keep in mind this will translate the molecule a bit due to 
        angles = np.random.uniform(0, [self.x, self.y, self.z])
        # Rotate the molecule around the center of mass
        rotation_matrix = self._get_rotation_matrix(angles)
        new_coords = (coords - center) @ rotation_matrix + center
        # Set the new coordinates to the conformer~
        conf.SetPositions(new_coords)
        return mol
    

class RandomFlipMolTransform:
    """
    Randomly flip a molecule around its center of mass.
    """
    def __init__(self, planes: tuple=(True, True, True)):
        """
        Initialize the transform with the given axes.
        For a full flip, with 50/50 probability for each plane, set all planes to True (Default).
        WARNING: the center of mass is not the same as the center of the bounding box.
        This may cause the molecule to be translated a bit. An additional alignment translation may compensate for this.

        Parameters
        ----------
        planes : tuple
            The planes to flip around, in (x=0, y=0, z=0) format. True means possible flip, False means no flip.
            Default is (True, True, True).
        """
        self.planes = planes

    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        # Get the geometric center of the molecule
        conf = mol.GetConformer(0)
        coords = conf.GetPositions()
        center = np.mean(coords, axis=0)
        probabilities = np.random.uniform(0, 1, size=3)
        flip_axes = [bool(self.planes[i] and prob > 0.5) for i, prob in enumerate(probabilities)]
        for plane in range(3):
            if flip_axes[plane]:
                coords[:, plane] = -coords[:, plane] + 2 * center[plane]
        # Set the new coordinates to the conformer
        conf.SetPositions(coords)
        return mol
                
        
        
def convert_xyz_to_sdf(input_path, output_sdf=None):
    """
    Converts from atom coordinates (.xyz supported) to .sdf, using OpenBabel.
    
    OpenBabel warnings are suppressed unless logging is set to DEBUG level.
    """
    if output_sdf is None:
        base_name = os.path.basename(os.path.normpath(input_path))
        output_sdf = p2m.TEMP_DIR / f"{base_name}_converted.sdf"
    
    # Ensure we pass a string path to pybel (OpenBabel expects a std::string)
    output_sdf = str(output_sdf)
    
    if os.path.isdir(input_path):
        # Process all .xyz files in the directory
        xyz_files = glob("*.xyz", root_dir=input_path)
        xyz_files = [os.path.join(input_path, f) for f in xyz_files]
    elif input_path.lower().endswith('.xyz'):
        xyz_files = [input_path]
    else:
        raise ValueError("input_path is not a directory or a xyz file.")
    
    # Use context manager to suppress OpenBabel warnings in non-verbose mode
    with suppress_openbabel_warnings():
        output = pybel.Outputfile("sdf", output_sdf, overwrite=True)
        
        for xyz_file in xyz_files:
            try:
                mol = next(pybel.readfile("xyz", xyz_file))
                mol.title = "end"  # Match VoxMol's behavior
                output.write(mol)
            except StopIteration:
                logger.warning(f"Could not read {xyz_file}")
            except Exception as e:
                logger.error(f"Error processing {xyz_file}: {e}")
        
        output.close()
    
    return output_sdf



    