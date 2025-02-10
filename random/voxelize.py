import numpy as np #TODO: switch to torch or cupy
from rdkit import Chem
import os
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def _calculate_voxels(channel_grid, coords, mode="binary"):
    """Calculate the voxels for a channel."""
    func_map = {
        "binary": _binary,
        "ivd": _inverse_squared_distance
    }
    try:
        func = func_map[mode]
    except KeyError:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {list(func_map.keys())}")
    
    shape = channel_grid.shape
    channel_grid = func(shape, coords)

    # print(channel_grid)
    
    
    return channel_grid


def _binary(shape, coords):
    grid = np.zeros(shape, dtype=np.float32)
    #get the indexes for each of the points
    indexes = np.floor(coords).astype(int)
    # print(indexes)
    #set the indexes to 1 and leave the rest as zeros
    grid[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = 1 #how does this even work with negative indexes?
    return grid

def _inverse_squared_distance(shape, coords):
    grid = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]): #not very effcient, but it works
        for j in range(shape[1]):
            for k in range(shape[2]):
                grid[i, j, k] = np.sum(1 / np.linalg.norm(coords - np.array([i, j, k]), axis=1) ** 2)
    return grid


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
    #translation to the origin step
    coords -= np.min(coords, axis=0)

    #DO NOT USE THE MOL OBJECT ANYMORE, IT HAS NOT BEEN UPDATED

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


    # Fill the grid
    for channel in CHANNELS:
        mask = np.array([feature == channel for feature in feature_types])
        channel_coords = coords[mask]
        if len(channel_coords) == 0:
            continue
        
        grid[CHANNELS[channel]] = _calculate_voxels(grid[CHANNELS[channel]], channel_coords, mode=mode)
        
        
    return grid



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    #load a mol
    datafile = "../data/zinc3d_test.sdf"
    suppl = Chem.SDMolSupplier(datafile, removeHs=False)
    mol = suppl[0]
    voxels = voxelize(mol, mode="ivd")
    print(voxels.shape)
    print(voxels[0])


    #plotting

    channel = 1 #carbon
    voxel_grid = voxels[channel]
    print(voxel_grid.shape)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ###### Plotting atoms and bonds ######
    atom_colors = {
        "C": "green",  # Carbon
        "O": "red",    # Oxygen
        "H": "gray",  # Hydrogen
        "N": "blue",   # Nitrogen
        "S": "yellow", # Sulfur
    }

    conf = mol.GetConformer()
    atom_positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    # Center molecule in the voxel grid
    min_bounds = atom_positions.min(axis=0)
    atom_positions -= min_bounds  # Shift so min coordinate starts at (0,0,0)

    #Plot atoms
    for i, pos in enumerate(atom_positions):
        atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()  # Get element type (C, O, H, etc.)
        color = atom_colors.get(atom_symbol, "white")  # Default to gray if unknown
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, label=atom_symbol if i == 0 else "")

    #Plot bonds
    bonds = []
    for bond in mol.GetBonds():
        start = atom_positions[bond.GetBeginAtomIdx()]
        end = atom_positions[bond.GetEndAtomIdx()]
        bonds.append([start, end])

    bond_collection = Line3DCollection(bonds, colors="black", linewidths=2)
    ax.add_collection3d(bond_collection)

    #get the size of the grid
    ax.set_xlim(0, voxel_grid.shape[0])
    ax.set_ylim(0, voxel_grid.shape[1])
    ax.set_zlim(0, voxel_grid.shape[2])
    ax.set_box_aspect([1, 1, 1])
    
    ax.voxels(voxel_grid, edgecolor="k", alpha=0.5)


    def set_axes_equal(ax):
        """Make the axes of a 3D plot have equal scale so voxels appear cubic."""
        extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        sizes = extents[:, 1] - extents[:, 0]
        max_size = max(sizes)
        centers = np.mean(extents, axis=1)
        
        ax.set_xlim(centers[0] - max_size / 2, centers[0] + max_size / 2)
        ax.set_ylim(centers[1] - max_size / 2, centers[1] + max_size / 2)
        ax.set_zlim(centers[2] - max_size / 2, centers[2] + max_size / 2)

    set_axes_equal(ax)  # Apply the fix


    plt.show()