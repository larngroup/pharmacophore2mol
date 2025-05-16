import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from rdkit import Chem

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


def get_translation_vector(points: np.ndarray) -> np.ndarray:
    """
    Get the translation vector to translate the points to the origin.
    This is useful to translate the points back to their original coordinates after voxelization.
    """
    if points.shape[1] != 3 or len(points.shape) != 2:
        raise ValueError(f"Points should be a 2d array with shape (#points, 3), but got {points.shape}")
    return - np.min(points, axis=0) #TODO: maybe add padding here?


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

    