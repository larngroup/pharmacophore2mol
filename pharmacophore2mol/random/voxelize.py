import numpy as np #TODO: switch to torch or cupy
from rdkit import Chem
import os
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def _calculate_voxels(channel_grid, coords, mode="binary", resolution=0.5):
    """Calculate the voxels for a channel."""
    func_map = {
        "binary": _binary,
        "ivd": _inverse_squared_distance,
        "gaussian": _gaussian
    }
    try:
        func = func_map[mode]
    except KeyError:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {list(func_map.keys())}")
    
    shape = channel_grid.shape
    channel_grid = func(shape, coords, l=resolution)

    # print(channel_grid)
    
    
    return channel_grid


def _binary(shape, coords, l=1):
    grid = np.zeros(shape, dtype=np.float32)
    #get the indexes for each of the points
    #if l=1, then indexes are just floor of the coords. if not, scaling seems a good idea
    coords = coords / l #TODO: check if this is correct
    indexes = np.floor(coords).astype(int)
    # print(indexes)
    #set the indexes to 1 and leave the rest as zeros
    grid[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = 1 #how does this even work with negative indexes?
    return grid

def _inverse_squared_distance(shape, coords, l=1):
    offset_to_center = l/2
    coords = coords / l
    grid = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]): #not very effcient, but it works
        for j in range(shape[1]):
            for k in range(shape[2]):
                #TODO: pass this through a sigmoid?
                grid[i, j, k] = np.sum(1 / np.linalg.norm(coords - np.array([i + offset_to_center, j + offset_to_center, k + offset_to_center]), axis=1) ** 2)
    return grid


def _gaussian(shape, coords, l=1):
    offset_to_center = l/2
    coords = coords / l
    std = 1
    scaled_std = std / l
    grid = np.zeros(shape, dtype=np.float32)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                grid[x, y, z] = np.max([np.exp(-np.linalg.norm(coords - np.array([x + offset_to_center, y + offset_to_center, z + offset_to_center]), axis=1) ** 2 / (2 * scaled_std ** 2))])
    return grid


def voxelize(mol: Chem.Mol, mode="binary", resolution = 0.5) -> np.ndarray:
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
    coords -= np.min(coords, axis=0) #TODO: add offset to center min atom on the grid or let it be?

    #DO NOT USE THE MOL OBJECT ANYMORE, IT HAS NOT BEEN UPDATED

    # Get the atomic numbers
    feature_types = [feature.GetSymbol() for feature in mol.GetAtoms()]
    # Get the min and max coordinates
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    # Calculate the grid size
    grid_size = np.ceil((max_coords - min_coords) / resolution).astype(int)
    # print(grid_size)
    # Create the grid
    grid = np.zeros((len(CHANNELS), *grid_size), dtype=np.float32)


    # Fill the grid
    for channel in CHANNELS:
        mask = np.array([feature == channel for feature in feature_types])
        channel_coords = coords[mask]
        if len(channel_coords) == 0:
            continue
        
        grid[CHANNELS[channel]] = _calculate_voxels(grid[CHANNELS[channel]], channel_coords, mode=mode, resolution=resolution)
        
        
    return grid


# def _plot_voxel_grid_and_mol(voxels: np.ndarray, mol: Chem.Mol, channel: int, resolution=0.5):
#     voxel_grid = voxels[channel]
#     print(voxel_grid.shape)
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection="3d")

#     ###### Plotting atoms and bonds ######
#     atom_colors = {
#         "C": "green",  # Carbon
#         "O": "red",    # Oxygen
#         "H": "gray",  # Hydrogen
#         "N": "blue",   # Nitrogen
#         "S": "yellow", # Sulfur
#     }

#     conf = mol.GetConformer()
#     atom_positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

#     # Center molecule in the voxel grid
#     min_bounds = atom_positions.min(axis=0)
#     atom_positions -= min_bounds  # Shift so min coordinate starts at (0,0,0)

#     #Plot atoms
#     for i, pos in enumerate(atom_positions):
#         atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()  # Get element type (C, O, H, etc.)
#         color = atom_colors.get(atom_symbol, "white")  # Default to gray if unknown
#         ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, label=atom_symbol if i == 0 else "")

#     #Plot bonds
#     bonds = []
#     for bond in mol.GetBonds():
#         start = atom_positions[bond.GetBeginAtomIdx()]
#         end = atom_positions[bond.GetEndAtomIdx()]
#         bonds.append([start, end])

#     bond_collection = Line3DCollection(bonds, colors="black", linewidths=2)
#     ax.add_collection3d(bond_collection)

#     #get the size of the grid
#     ax.set_xlim(0, voxel_grid.shape[0])
#     ax.set_ylim(0, voxel_grid.shape[1])
#     ax.set_zlim(0, voxel_grid.shape[2])
#     ax.set_box_aspect([1, 1, 1])
    
#     ax.voxels(voxel_grid, edgecolor="k", alpha=0.5)


#     extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
#     sizes = extents[:, 1] - extents[:, 0]
#     max_size = max(sizes)
#     centers = np.mean(extents, axis=1)
    
#     ax.set_xlim(centers[0] - max_size / 2, centers[0] + max_size / 2)
#     ax.set_ylim(centers[1] - max_size / 2, centers[1] + max_size / 2)
#     ax.set_zlim(centers[2] - max_size / 2, centers[2] + max_size / 2)



#     plt.show()


if __name__ == "__main__1":
    import matplotlib.pyplot as plt
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    #load a mol
    datafile = "../data/zinc3d_test.sdf"
    suppl = Chem.SDMolSupplier(datafile, removeHs=False)
    mol = suppl[0]
    voxels = voxelize(mol, mode="binary")
    print(voxels.shape)
    print(voxels[0])


    #plotting
    # _plot_voxel_grid_and_mol(voxels, mol, 0)
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sdf = '''lig.pdb


 28 31  0  0  0  0  0  0  0  0999 V2000
  -12.3750   15.6630   41.2650 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2610   14.3660   39.8260 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.1090   14.4140   41.0570 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.4740   15.2840   42.0050 N   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1510   15.2100   40.0160 N   0  0  0  0  0  0  0  0  0  0  0  0
  -16.6176   14.1407   40.7369 C   0  0  0  0  0  0  0  0  0  0  0  0
  -15.3520   13.6930   41.3060 C   0  0  0  0  0  0  0  0  0  0  0  0
  -16.4718   11.9054   42.3009 C   0  0  0  0  0  0  0  0  0  0  0  0
  -17.7694   13.3442   41.0488 C   0  0  0  0  0  0  0  0  0  0  0  0
  -15.3321   12.5656   42.0833 N   0  0  0  0  0  0  0  0  0  0  0  0
  -17.6782   12.2407   41.8231 N   0  0  0  0  0  0  0  0  0  0  0  0
  -16.3982   10.8282   43.0485 N   0  0  0  0  0  0  0  0  0  0  0  0
  -17.2100   10.3051   43.2395 H   0  0  0  0  0  0  0  0  0  0  0  0
  -15.5311   10.5438   43.4180 H   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2773   14.5917   44.3887 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5557   14.9516   45.8687 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.4526   17.4692   45.3119 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2634   17.1035   43.8042 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.8090   15.6920   43.4140 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.0728   16.3354   46.2392 N   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0543   16.3036   46.2844 H   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2602   13.6411   37.6205 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2997   12.7333   36.5337 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.4040   11.6421   36.4989 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4709   11.4518   37.5318 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4272   12.3524   38.6174 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.3220   13.4750   38.6950 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.4377   10.7705   35.4697 F   0  0  0  0  0  0  0  0  0  0  0  0
 15 16  1  0  0  0
 15 19  1  0  0  0
 16 20  1  0  0  0
 17 18  1  0  0  0
 17 20  1  0  0  0
 18 19  1  0  0  0
 19  4  1  0  0  0
 22 23  2  0  0  0
 22 27  1  0  0  0
 23 24  1  0  0  0
 24 25  2  0  0  0
 24 28  1  0  0  0
 25 26  1  0  0  0
 26 27  2  0  0  0
 27  2  1  0  0  0
  1  5  2  0  0  0
  1  4  1  0  0  0
  2  3  2  0  0  0
  2  5  1  0  0  0
  3  4  1  0  0  0
  3  7  1  0  0  0
  6  7  2  0  0  0
  6  9  1  0  0  0
  7 10  1  0  0  0
  8 11  1  0  0  0
  8 12  1  0  0  0
  8 10  2  0  0  0
  9 11  2  0  0  0
 20 21  1  0  0  0
 12 13  1  0  0  0
 12 14  1  0  0  0
M  END
> <minimizedAffinity>
0.00000

> <minimizedRMSD>
0.64667

$$$$'''

    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdf, removeHs=False)
    mol = suppl[0]
    voxel_grid = voxelize(mol, mode="gaussian", resolution=0.22)
    plt.matshow(voxel_grid[0,22,:,:])
    plt.show()

