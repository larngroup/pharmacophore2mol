import numpy as np
import time

def extract_subgrids_with_numpy(voxel_grid, important_voxels, side, stride, return_subgrids=False):
    """
    Extracts subgrids of a given size and stride from the 3D voxel grid,
    keeping only subgrids that contain at least one important voxel.
    
    Parameters:
        voxel_grid (np.ndarray): The original 3D voxel grid (numpy array).
        important_voxels (np.ndarray): Array of important voxel indices (num_important, 3).
        side (int): The size of the cubic subgrid.
        stride (int): The stride between subgrids.
        return_subgrids (bool): If True, returns the subgrids themselves, otherwise the "lowest corners".
    
    Returns:
        list: A list of either subgrids or their lowest corner coordinates.
    """
    grid_size_x, grid_size_y, grid_size_z = voxel_grid.shape
    
    # List to store either the subgrids or their lowest corner coordinates
    results = []

    # Iterate over the grid with the given stride
    for x in range(0, grid_size_x - side + 1, stride):
        for y in range(0, grid_size_y - side + 1, stride):
            for z in range(0, grid_size_z - side + 1, stride):
                # Define the boundaries of the current subgrid
                x_end, y_end, z_end = x + side, y + side, z + side
                
                # Check if any important voxel is inside the current subgrid
                mask = (important_voxels[:, 0] >= x) & (important_voxels[:, 0] < x_end) & \
                       (important_voxels[:, 1] >= y) & (important_voxels[:, 1] < y_end) & \
                       (important_voxels[:, 2] >= z) & (important_voxels[:, 2] < z_end)

                if np.any(mask):  # If any important voxel is in the subgrid
                    # Store the lowest corner (x, y, z) of the subgrid
                    results.append((x, y, z))
    
    return results


def extract_subgrids_with_point_cloud_expansion(voxel_grid, important_voxels, side, stride, return_subgrids=False):
    """
    Same as before but with a different approach to extract the subgrids.
    Starts by looking at the important voxels and creates a set of all the lowest corners that contain each of them, appending them to the results.
    Then, removes the duplicates and returns the list of lowest corners.
    """
    grid_size_x, grid_size_y, grid_size_z = voxel_grid.shape

    results = []

    # Iterate over the important voxels
    for i, j, k in important_voxels:
        pass
    
    
    results = []

    # Iterate over the important voxels
    for i, j, k in important_voxels:
        min_x = max(0, i - side + 1)
        min_y = max(0, j - side + 1)
        min_z = max(0, k - side + 1)

        max_x = min(grid_size_x - side + 1, i + 1) #MAX IS EXCLUSIVE!!!
        max_y = min(grid_size_y - side + 1, j + 1)
        max_z = min(grid_size_z - side + 1, k + 1)

        lowest_corners = expand_ranges((min_x, max_x), (min_y, max_y), (min_z, max_z))
        results.append(lowest_corners)

    results = np.vstack(results)
    # Remove duplicates
    results = np.unique(results, axis=0)

    return results




def expand_ranges(x, y, z):
    """
    Expand the ranges of the 3D space into a list of coordinates.
    x, y and z are tuples like (min, max) (min included, max excluded).
    Returns a list of coordinates (x, y, z).
    
    Parameters
    ----------
    x : tuple
        The range of the x-axis.
    y : tuple
        The range of the y-axis.
    z : tuple
        The range of the z-axis.

    Returns
    -------
    np.ndarray
        A 2D array with all the coordinates in the 3D space.
    """
    x_min, x_max = x
    y_min, y_max = y
    z_min, z_max = z
    # Define the ranges for each axis
    x_range = np.arange(x_min, x_max)
    y_range = np.arange(y_min, y_max)
    z_range = np.arange(z_min, z_max)

    # Create meshgrid for the 3D space
    x, y, z = np.meshgrid(x_range, y_range, z_range)

    # Stack them into a single 2D array (each row is a point in 3D space)
    coordinates = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Now, 'coordinates' contains all the possible (x, y, z) points
    return coordinates

# Example 4x4x4 voxel grid
voxel_grid = np.zeros((50, 56, 57))

# Some important voxel coordinates
important_voxels = np.array([
    [0, 1, 2],
    [3, 3, 0],
    [1, 2, 3],
    [3, 5, 7],
    [15, 8, 4],
    [19, 14, 0],
    [10, 2, 15],
    [0, 12, 3],
    [13, 1, 16],
    [6, 10, 11],
    [8, 4, 2],
    [12, 15, 14],
    [17, 3, 9],
    [5, 6, 8],
    [2, 13, 1],
    [11, 7, 16],
    [9, 12, 5],
    [4, 1, 0]
])

# voxel_grid = np.zeros((20, 20, 20))

# # Some important voxel coordinates
# important_voxels = np.array([
#     [0, 1, 2],
#     [3, 3, 0],
#     [1, 2, 3],
#     [3, 3, 3]
# ])

side = 2
stride = 1
# Fragment the grid with side=2, stride=1
start_time = time.time()
lowest_corners = extract_subgrids_with_numpy(voxel_grid, important_voxels, side=side, stride=stride, return_subgrids=False)
end_time = time.time()

print("Execution time:", end_time - start_time, "seconds")
print("Nr of possible subgrids:", voxel_grid.shape[0] * voxel_grid.shape[1] * voxel_grid.shape[2])
print("Nr of filtered subgrids:", len(lowest_corners))
# print(lowest_corners)

start_time = time.time()
lowest_corners = extract_subgrids_with_point_cloud_expansion(voxel_grid, important_voxels, side=side, stride=stride, return_subgrids=False)
end_time = time.time()

print("Execution time:", end_time - start_time, "seconds")
print("Nr of possible subgrids:", voxel_grid.shape[0] * voxel_grid.shape[1] * voxel_grid.shape[2])
print("Nr of filtered subgrids:", len(lowest_corners))
# print(lowest_corners)

# print("Nr of possible subgrids:", voxel_grid.shape[0] * voxel_grid.shape[1] * voxel_grid.shape[2])
# print("Nr of filtered subgrids:", len(lowest_corners))
# print(lowest_corners)


# coords1 = expand_ranges((0, 50), (0, 30), (0, 20))
# coords2 = expand_ranges((30, 50), (25, 30), (0, 20))
# coords3 = expand_ranges((0, 10), (5, 15), (10, 20))

# all_coords = np.vstack([coords1, coords2, coords3])
# print(all_coords.shape)

# unique_coords = np.unique(all_coords, axis=0)
# print(unique_coords.shape)

