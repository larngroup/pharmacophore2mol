import numpy as np #TODO: switch to torch or cupy
from rdkit import Chem
import os


def _calculate_voxels(channel_grid, coords, mode="binary"):
    """Calculate the voxels for a channel."""
    func_map = {
        "binary": _binary
    }
    try:
        func = func_map[mode]
    except KeyError:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {list(func_map.keys())}")
    
    
    return channel_grid


def _binary(x, y, z, coords):
    return 1


def voxelize(mol: Chem.Mol, mode="binary") -> np.ndarray:
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
        "Si": 10,
    }
    # Get the coordinates
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    # Get the atomic numbers
    feature_types = [feature.GetSymbol() for feature in mol.GetAtoms()]
    # Get the min and max coordinates
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    # Calculate the grid size
    grid_size = np.ceil(max_coords - min_coords).astype(int) + 1
    # print(grid_size)
    # Create the grid
    grid = np.zeros((len(CHANNELS), *grid_size), dtype=np.float32)

    # def binary(mask, coords):
        


    # grid = np.apply_along_axis(binary, -1, distance_grid, coords=coords)
    # print(grid.shape)
    # print(grid[0, 0, 0, 0])

    # Fill the grid
    for channel in CHANNELS:
        mask = np.array([feature == channel for feature in feature_types])
        channel_coords = coords[mask]
        if len(channel_coords) == 0:
            continue
        
        grid[CHANNELS[channel]] = _calculate_voxels(grid[CHANNELS[channel]], channel_coords, mode="binary")
        
        
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