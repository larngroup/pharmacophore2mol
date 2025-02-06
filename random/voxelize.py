import numpy as np #TODO: switch to torch or cupy
from rdkit import Chem
import os

def voxelize(mol: Chem.Mol) -> np.ndarray:
    """Voxelize a molecule."""
    CHANNELS = { #keeping this as a dict instead of a list for hashtable lookup
        "C": 0,
        "H": 1,
        "N": 2,
        "O": 3,
        "F": 4,
        "P": 5,
        "S": 6,
        "Cl": 7,
        "Br": 8,
        "I": 9,
    }
    # Get the coordinates
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    # Get the atomic numbers
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # Get the min and max coordinates
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    # Calculate the grid size
    grid_size = np.ceil(max_coords - min_coords).astype(int) + 1
    print(grid_size)
    # Create the grid
    grid = np.zeros(grid_size, dtype=np.float32)

    #create a distance matrix for each atom to each voxel center (going full pairwise here, if too slow swith to a KDTree with nearest neighbours)
    distance_matrix = np.zeros((*grid_size, len(coords)), dtype=np.float32)
    for i, coord in enumerate(coords):
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    distance_matrix[x, y, z, i] = np.linalg.norm(coord - np.array([x, y, z]) - min_coords)

    # Fill the grid
    for channel in CHANNELS:
        #get the relevant distance matrices
        channel_indices = [i for i, atom in enumerate(atom_types) if atom == channel]
        channel_distance_matrix = distance_matrix[channel_indices]
        print(channel_distance_matrix.shape)
        print(channel_distance_matrix.T)
    return grid



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    #load a mol
    datafile = "../data/zinc3d_test.sdf"
    suppl = Chem.SDMolSupplier(datafile, removeHs=False)
    mol = suppl[0]
    voxels = voxelize(mol)
    print(voxels.shape)
    # plt.imshow(voxels.sum(axis=0))
    # plt.show()