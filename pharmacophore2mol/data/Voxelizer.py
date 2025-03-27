"""
Voxelizer
---------
This module contains the Voxelizer class, which is used to voxelize a point cloud, like some molecular representations or pharmacophores
It goes with a fixed resiolution approach instead of a fixed grid size approach.
It also contains some functionality intended to fragment the voxel grid into smaller grids.
"""


import numpy as np
from math import ceil

class Voxelizer:
    """
    This class is intended to be used to voxelize a point cloud, like some molecular representations or pharmacophores.
    It goes with a fixed resiolution approach instead of a fixed grid size approach.
    This means that, without proper padding, the output will be variable sized voxel grids.
    """
    def __init__(self, channels: list, resolution: float, mode="gaussian"):
        """
        Initialize the Voxelizer object.

        Parameters
        ----------
        channels: list
            A list with the names of the channels to be used. The order of the names will be used to
            index the channels in the resulting grid.
        resolution: float
            The resolution of the voxel grid. It will be used to calculate the size of the grid and
            the indexes of the points in the grid.
        mode: str, optional
            The mode to use when calculating the voxels. Available modes are: "binary", "ivd" and "gaussian".
            The "binary" mode will set the voxels to 1 if a point is present in it, and 0 otherwise.
            The "gaussian" mode will calculate the value for each voxel as f(distance_to_point(p))
            , where f is a gaussian function centered at 0 and standard deviation of 1, and choose
            the maximum value for all points p.
        """
        self.channels = {c: i for i, c in enumerate(channels)}
        self.mode = mode
        self.resolution = resolution

    def __repr__(self):
        return f"Voxelizer(channels={self.channels}, resolution={self.resolution}, mode={self.mode})"
    
    def get_channels(self):
        return self.channels
    
    def voxelize(self, points: dict, min_grid_size: tuple | None = None, allow_negative_coords=False) -> np.ndarray: #python 3.10+ stuff, not that portable but pytorch already kinda limits backport to earlier python
        """
        Voxelize a point cloud.

        Parameters
        ----------
        points: dict
            A dictionary with the channel name as key and the points' coordinates as value.
            The points should be a 2d array, with shape (#points, 3).
            Example::

                {
                    'channel_1': np.array([
                        [1, 2, 3],
                        [1, 5, 3]
                    ]),
                    'channel_2': np.array([
                        [2, 1, 5]
                    ])
                }

        min_grid_size: tuple or None, optional
            A tuple with the minimum voxel grid dimensions. If None, the the minimum size to fit all the
            points will be automatically calculated. Dimentions should be passed as absolute values
            (meaning in the same units as the points), and not as the relative size according to the
            voxel resolution. The dimentions shall be interpreted as minimum values, and will be rounded
            up to the nearest multiple of the resolution.
            For example, if a box_size of (10, 10, 10) is passed with a resolution of 0.75, the actual
            box size will be rounded up to (10.5, 10.5, 10.5). The extra 0.5 padding will be added only to
            the maximum sides of the axes of the reference frame, meaning no centering will be made.
        allow_negative_coords: bool, optional
            If True, negative coordinates will be allowed, but keep in mind they will be translated to origin. If False, will raise an error if any negative
            coordinates are found in the points parameter.
        """

        #check if translation should be forced
        if not allow_negative_coords:
            all_coords = [point for channel in points for point in points[channel]]
            # print(all_coords)
            if min([min(point) for point in all_coords]) < 0:
                raise ValueError("Negative coordinates found. If you want to forcefully allow negative coordinates, set allow_negative_coords=True.")

        else:
            raise NotImplementedError("Negative coordinates need translation to origin. NYI.")
        
        
        #calculate the minimum grid size and initialize
        if min_grid_size is None:
            min_grid_size = np.array([max([max([point[i] for point in points[channel]]) for channel in points]) for i in range(3)])
        elif not isinstance(min_grid_size, np.ndarray):
            min_grid_size = np.array(min_grid_size)
        
        if min_grid_size.shape != (3,):
            raise ValueError(f"min_grid_size should be a 3d tuple or list, but got {min_grid_size.shape}")

        grid_shape = np.floor(min_grid_size / self.resolution).astype(int) + 1
        # print(grid_shape)
        grid = np.zeros((len(self.channels), *grid_shape), dtype=np.float32)
        
        #actually fill the grid
        for c in self.channels:
            channel_coords = points.get(c, np.empty((0)))
            if len(channel_coords) == 0:
                continue
            
            grid[self.channels[c]] = self._calculate_voxels(grid[self.channels[c]], channel_coords)

        return grid

    def _calculate_voxels(self, channel_grid: np.ndarray, coords: np.ndarray | list):
        """Calculate the voxels for a channel."""
        func_map = {
            "binary": self._binary,
            "ivd": self._inverse_squared_distance,
            "gaussian": self._gaussian
        }

        if isinstance(coords, list):
            try: #try to convert to np array
                coords = np.array(coords)
            except Exception as e:
                raise ValueError(f"Could not convert coords to a 2d numpy array. Please make sure the format it was passed in is convertible, like a list of lists, list of tuples, etc")


        try:
            func = func_map[self.mode]
        except KeyError:
            raise ValueError(f"Invalid mode: {self.mode}. Available modes: {list(func_map.keys())}. If you just added a new mode, please don't forget to name it and add it to the Voxelizer._calculate_voxels method")
        
        shape = channel_grid.shape
        channel_grid = func(shape, coords)
        return channel_grid
    
    def _binary(self, shape, coords: np.ndarray):
        grid = np.zeros(shape, dtype=np.float32)
        #get the indexes for each of the points
        #if l=1, then indexes are just floor of the coords. if not, scaling seems a good idea
        indexes = self.get_indexes(coords)
        # print(indexes)
        #set the indexes to 1 and leave the rest as zeros
        grid[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = 1 #how does this even work with negative indexes?
        return grid

    def _inverse_squared_distance(self, shape, coords: np.ndarray):
        raise DeprecationWarning("_inverse_square_distance: This method was abandoned and may contain bugs.")
        offset_to_center = self.resolution/2
        coords = coords / self.resolution
        grid = np.zeros(shape, dtype=np.float32)
        for i in range(shape[0]): #not very effcient, but it works
            for j in range(shape[1]):
                for k in range(shape[2]):
                    #TODO: pass this through a sigmoid?
                    grid[i, j, k] = np.sum(1 / np.linalg.norm(coords - np.array([i + offset_to_center, j + offset_to_center, k + offset_to_center]), axis=1) ** 2)
        return grid

    def _gaussian(self, shape, coords: np.ndarray):
        offset_to_center = self.resolution/2
        coords = coords / self.resolution
        std = 1
        scaled_std = std / self.resolution
        grid = np.zeros(shape, dtype=np.float32)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    grid[x, y, z] = np.max([np.exp(-np.linalg.norm(coords - np.array([x + offset_to_center, y + offset_to_center, z + offset_to_center]), axis=1) ** 2 / (2 * scaled_std ** 2))])
        return grid
    
    def get_indexes(self, coords: np.ndarray) -> np.ndarray:
        """
        Get the indexes that contain the points specified in the coords parameter.
        """
        coords = coords / self.resolution
        return np.floor(coords).astype(int)
    
    def distance_to_voxel(self, distance: float) -> int:
        """
        Convert a distance (in whatever units the voxelizer is working with) to
        the nearest corresponding number of voxels, rounded up.
        """
        return ceil(distance / self.resolution)
    



def fragment_voxel_grid(grid: np.ndarray, side: int, stride: int=1, roi_indices: np.ndarray=None) -> np.ndarray:
    """
    Fragment a voxel grid into smaller grids, cubic, fixed size, possibly overlapping grids.
    It fragments the grid into smaller grids of size side x side x side, with a stride of stride.
    (Therefore, if stride < side, the grids will overlap, as it is intentional sometimes)
    If stride is None, it will default to side.
    If roi_indices (Region of Interest Indices) is passed, it will drop the fragments that do not contain
    at least one of such indices.

    Parameters
    ----------
    grid: np.ndarray
        The voxel grid to fragment. It should be a 4d array with shape (channels, x, y, z).
    side: int
        The size of the cubic fragments, in voxels.
    stride: int or None, optional
        The stride between fragments. If None, it will default to side, in voxels.
    roi_indices: np.ndarray, optional
        A 2d array with the coordinates of the points of interest. It should have shape (#points, 3).
        If None, no subgrids will be dropped.

    Returns
    -------
    np.ndarray
        A 5d array with shape (num_fragments, channels, side, side, side) containing the fragments.
    """


    if roi_indices is not None:
        assert roi_indices.shape[1] == 3, "Point cloud coordinates should be a 2d array with shape (#points, 3)"
        roi_indices = roi_indices.astype(int)
        if len(roi_indices) == 0:
            raise ValueError("roi_indices should have at least one point")
        if (roi_indices < 0).any():
            raise ValueError("roi_indices should not contain negative coordinates")
        
        low_corners = _get_low_corners(grid.shape[1:], roi_indices, side, stride)
    
    else:
        max_x, max_y, max_z = [ceil((dim_size - side + 1) / stride) * stride for dim_size in grid.shape[1:]] #MAX IS EXCLUSIVE!!!
        low_corners = _expand_ranges((0, max_x), (0, max_y), (0, max_z), step=stride)

    fragments = []
    for x, y, z in low_corners:
        fragment = grid[:, x:x+side, y:y+side, z:z+side]
        fragments.append(fragment)

    return np.array(fragments)
        
    



def _get_low_corners(voxel_grid_shape: tuple, roi_indices: np.ndarray, side: int, stride: int) -> np.ndarray:
    """
    Get the lowest corners of the subgrids that contain at least one of the important voxels.
    "Lowest corner" is the index (x, y, z) of the voxel with lowest x, y, z coordinates in the subgrid.

    Parameters
    ----------
    voxel_grid_shape: tuple
        The shape of the voxel grid. Should be a tuple with the shape (x, y, z).
        Length should be 3, not 4, as it is a shape, not a grid.
    roi_indices: np.ndarray
        The indices of the important voxels. Should be a 2D array with shape (#points, 3).
    side: int
        The size of the cubic fragments, in voxels.
    stride: int
        The stride between fragments, in voxels.
    """

    grid_size_x, grid_size_y, grid_size_z = voxel_grid_shape

    
    results = []

    # Iterate over the important voxels
    for i, j, k in roi_indices:
        min_x = ceil(max(0, i - side + 1) / stride) * stride # USE THE CEIL FROM MATH!! for non array operations, math module is 10x faster than numpy
        min_y = ceil(max(0, j - side + 1) / stride) * stride
        min_z = ceil(max(0, k - side + 1) / stride) * stride
        

        max_x = ceil(min(grid_size_x - side + 1, i + 1) / stride) * stride #MAX IS EXCLUSIVE!!!
        max_y = ceil(min(grid_size_y - side + 1, j + 1) / stride) * stride
        max_z = ceil(min(grid_size_z - side + 1, k + 1) / stride) * stride
        lowest_corners = _expand_ranges((min_x, max_x), (min_y, max_y), (min_z, max_z), step=stride)
        results.append(lowest_corners)

    results = np.vstack(results)
    # Remove duplicates
    results = np.unique(results, axis=0)

    return results


def _expand_ranges(x, y, z, step=1):
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
    x_range = np.arange(x_min, x_max, step)
    y_range = np.arange(y_min, y_max, step)
    z_range = np.arange(z_min, z_max, step)

    # Create meshgrid for the 3D space
    x, y, z = np.meshgrid(x_range, y_range, z_range)

    # Stack them into a single 2D array (each row is a point in 3D space)
    coordinates = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Now, 'coordinates' contains all the possible (x, y, z) points
    return coordinates




if __name__ == "__main__":
    v = Voxelizer(channels=["C", "H", "N"], resolution=0.05, mode="gaussian")
    grid = v.voxelize({"C": [(0, 0, 0), (1, 1, 1)], "H": [(0.5, 0.5, 0.5)]}, (1, 1, 1))
    
    import matplotlib.pyplot as plt
    # plt.imshow(grid[0, 0, :, :])
    # plt.show()

    side = v.distance_to_voxel(0.3)
    stride = v.distance_to_voxel(0.1)
    roi_indices = v.get_indexes(np.array([(0.5, 0.5, 0.5)]))
    fragments = fragment_voxel_grid(grid, side, stride, roi_indices)
    print(fragments.shape)

    plt.imshow(fragments[4, 1, 0, :, :])
    plt.show()
    