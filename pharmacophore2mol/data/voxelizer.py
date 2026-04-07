"""
Voxelizer
---------
This module contains the Voxelizer class, which is used to voxelize a point cloud, like some molecular representations or pharmacophores
It goes with a fixed resiolution approach instead of a fixed grid size approach.
It also contains some functionality intended to fragment the voxel grid into smaller grids.
"""


import numpy as np
from math import ceil
from pharmacophore2mol.data.utils import get_translation_vector, mol_to_atom_dict
from scipy.spatial.distance import cdist

class Voxelizer:
    """
    This class is intended to be used to voxelize a point cloud, like some molecular representations or pharmacophores.
    It goes with a fixed resiolution approach instead of a fixed grid size approach.
    This means that, without proper padding, the output will be variable sized voxel grids.
    """
    def __init__(self, channels: list, resolution: float, mode="gaussian", **kwargs):
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
        self._channels = {c: i for i, c in enumerate(channels)}
        self.mode = mode
        self.resolution = resolution
        self.kwargs = kwargs

    def __repr__(self):
        return f"Voxelizer(channels={self._channels}, resolution={self.resolution}, mode={self.mode})"
    
    @property
    def channels(self):
        return self._channels
    
    @channels.setter
    def channels(self, channels: list):
        self._channels = {c: i for i, c in enumerate(channels)}

    
    def set_channels(self, channels: list):
        self.channels = channels
    
    def voxelize(self, points: dict, center: tuple | np.ndarray, side_length: float, force_shape: tuple | None = None) -> np.ndarray: #python 3.10+ syntax, but the backport is already limited by torch
        """
        Voxelize a point cloud within a specific cubic region of interest.

        Parameters
        ----------
        points: dict
            A dictionary with the channel name as key and the points' coordinates as value.
            The points should be a 2d array, with shape (#points, 3).
            Example: {'C': np.array([[1, 2, 3], ...]), ...}

        center : tuple or np.ndarray
            The spatial center (x, y, z) of the region to voxelize, in Angstroms.
            
        side_length : float
            The side length of the cubic region, in Angstroms. 
            The voxel grid will extend from `center - side/2` to `center + side/2`.
            
        force_shape: tuple, optional
            Force the output grid to have this specific shape (x, y, z), overriding the 
            calculated shape from side_length/resolution.
        """
        center_arr = np.asarray(center)
        if center_arr.shape != (3,):
            raise ValueError(f"Center must be a 3-element tuple or array. Got shape {center_arr.shape}")

        if force_shape is not None:
             grid_shape = np.array(force_shape)
             real_side_lengths = grid_shape * self.resolution
        else: #auto adjust, using ceil, meaning side length is AT LEAST the size covered by the grid
             dim_size = int(np.ceil(side_length / self.resolution))
             grid_shape = np.array([dim_size, dim_size, dim_size])
             real_side_lengths = np.array([dim_size * self.resolution] * 3)

        origin = center_arr - (real_side_lengths / 2.0) #this way center is center even after the adjust
        
        translation_vector = -origin #translating is just easier to math
            
        # shape is (C, D, H, W)
        grid = np.zeros((len(self._channels), *tuple(grid_shape)), dtype=np.float32)
        
        for c in self._channels:
            coords = points.get(c)
            if coords is None or len(coords) == 0:
                continue

            channel_coords = coords + translation_vector
            
            grid[self._channels[c]] = self._calculate_voxels(grid[self._channels[c]], channel_coords)

        return grid
    
    def get_min_grid_size(self, points: np.ndarray) -> np.ndarray:
        """
        Get the minimum grid size to fit all the points.
        It will be rounded up to the nearest multiple of the resolution.
        The extra padding will be added only to the maximum sides of the axes of the reference frame, meaning no centering will be made.

        Parameters
        ----------
        points: np.ndarray
            A 2d array with the coordinates of the points. It should have shape (#points, 3).
            
        
        Returns
        -------
        np.ndarray
            A 1d array with the minimum grid size. It will have shape (3,).
        """

        if points.shape[1] != 3 or len(points.shape) != 2:
            raise ValueError(f"Points should be a 2d array with shape (#points, 3), but got {points.shape}")
        maxes = np.max(points, axis=0)
        # #get the index of the maximum value for each axis
        # maxes_idx = self.get_indexes(maxes)

        return maxes
    
    

        

    def _calculate_voxels(self, channel_grid: np.ndarray, coords: np.ndarray | list):
        """
        Calculate the voxels for a channel.

        Parameters
        ----------
        channel_grid: np.ndarray
            The grid for the channel to be filled. It should have shape (x, y, z).
        coords: np.ndarray or list
            The coordinates of the points for the channel. It should have shape (#points, 3).
        
        Returns
        -------
        np.ndarray
            The grid for the channel, with the voxels filled according to the mode.
        """

        func_map = {
            "binary": self._binary,
            # "ivd": self._inverse_squared_distance,
            "gaussian": self._gaussian,
            "dry_run": self._dry_run
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
        channel_grid = func(shape, coords, **self.kwargs)
        return channel_grid
    
    def _binary(self, shape, coords: np.ndarray):
        grid = np.zeros(shape, dtype=np.float32)
        #get the indexes for each of the points
        #if l=1, then indexes are just floor of the coords. if not, scaling seems a good idea
        indexes = self.get_indexes(coords) #already handled resolution in the get_indexes method
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

    def _gaussian(self, shape, coords: np.ndarray, std: float=1.0, pooling="prob"):
        if pooling not in ["max", "avg", "sum", "prob"]:
            raise ValueError(f"Invalid pooling mode: {pooling}. Available modes: ['max', 'avg', 'sum', 'prob']")

        scaled_std = std / self.resolution
        variance_factor = -0.5 / (scaled_std ** 2)
        cutoff = 3.0 * scaled_std 
        
        coords_v = coords / self.resolution
        
        if pooling == "prob":
            grid_gauss = np.ones(shape, dtype=np.float32)
        else:
            grid_gauss = np.zeros(shape, dtype=np.float32)
            
        for point in coords_v:
            min_idx = np.maximum(0, np.floor(point - cutoff).astype(int))
            max_idx = np.minimum(shape, np.ceil(point + cutoff).astype(int) + 1)
            
            if np.any(min_idx >= shape) or np.any(max_idx <= 0):
                continue
                
            x_centers = np.arange(min_idx[0], max_idx[0]) + 0.5
            y_centers = np.arange(min_idx[1], max_idx[1]) + 0.5
            z_centers = np.arange(min_idx[2], max_idx[2]) + 0.5
            
            dx2 = (x_centers - point[0]) ** 2
            dy2 = (y_centers - point[1]) ** 2
            dz2 = (z_centers - point[2]) ** 2
            
            dist_sq = dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]
            local_gauss = np.exp(dist_sq * variance_factor)
            
            region = np.s_[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]
            
            if pooling == "max":
                grid_gauss[region] = np.maximum(grid_gauss[region], local_gauss)
            elif pooling == "sum":
                grid_gauss[region] += local_gauss
            elif pooling == "avg":
                grid_gauss[region] += local_gauss ** 2
            elif pooling == "prob": #apparently there's a voxelization library already, called pyuul. this is their method of pooling, which is actually pretty clever
                grid_gauss[region] *= (1.0 - local_gauss)

        if pooling == "prob":
            np.subtract(1.0, grid_gauss, out=grid_gauss)
            
        return grid_gauss
    
    def _dry_run(self, shape, coords: np.ndarray):
        """
        This method is just a dry run to simulate the voxelization calculations without actually doing it.
        Just returns a grid in the correct shape, but not initialized.
        """
        return np.empty(shape, dtype=np.float32)
    
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
        
        low_corners = _get_low_corners(grid.shape[1:], side, stride, roi_indices)
    
    else:
        max_x, max_y, max_z = [ceil((dim_size - side + 1) / stride) * stride for dim_size in grid.shape[1:]] #MAX IS EXCLUSIVE!!!
        low_corners = _expand_ranges((0, max_x), (0, max_y), (0, max_z), step=stride)
    
    fragments = []
    for x, y, z in low_corners:
        fragment = grid[:, x:x+side, y:y+side, z:z+side]
        fragments.append(fragment)

    return np.array(fragments)


def get_frag_count(grid_shape: tuple, side: int, stride: int, roi_indices: np.ndarray) -> int:
    """
    Get the number of fragments that will be generated from the voxel grid.

    Parameters
    ----------
    grid_shape: tuple
        The shape of the voxel grid. Should be a tuple with the shape (x, y, z).
        Length should be 3, not 4, as it is a shape, not a grid.
    side: int
        The size of the cubic fragments, in voxels.
    stride: int
        The stride between fragments, in voxels.
    roi_indices: np.ndarray
        The indices of the important voxels. Should be a 2D array with shape (#points, 3).
    """
    
    low_corners = _get_low_corners(grid_shape, side, stride, roi_indices)
    return len(low_corners)
    



def _get_low_corners(voxel_grid_shape: tuple, side: int, stride: int, roi_indices: np.ndarray) -> np.ndarray:
    """
    Get the lowest corners of the subgrids that contain at least one of the important voxels.
    "Lowest corner" is the index (x, y, z) of the voxel with lowest x, y, z coordinates in the subgrid.

    Parameters
    ----------
    voxel_grid_shape: tuple
        The shape of the voxel grid. Should be a tuple with the shape (x, y, z).
        Length should be 3, not 4, as it is a shape, not a grid.
    side: int
        The size of the cubic fragments, in voxels.
    stride: int
        The stride between fragments, in voxels.
    roi_indices: np.ndarray
        The indices of the important voxels. Should be a 2D array with shape (#points, 3).
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
    from pharmacophore2mol.data.utils import plot_voxel_grid_sweep
    import matplotlib.pyplot as plt
    from rdkit import Chem
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    suppl = Chem.SDMolSupplier("./raw/zinc3d_test.sdf", removeHs=False, sanitize=False, strictParsing=False)
    v = Voxelizer(channels=["C", "H", "N"], resolution=0.20, mode="gaussian", pooling="max", std=0.5) #defaults
    mol = suppl[0]
    atom_dict = mol_to_atom_dict(mol)

    # grid = v.voxelize({"C": [(0, 0, 0), (1, 1, 1)], "H": [(0.5, 0.5, 0.5)]}, (1, 1, 1))
    grid = v.voxelize(atom_dict, allow_negative_coords=True)
    
    # plt.imshow(grid[0, 10, :, :])
    # plt.show()
    plot_voxel_grid_sweep(grid[0])


    side = v.distance_to_voxel(0.3)
    stride = v.distance_to_voxel(0.1)
    # roi_indices = v.get_indexes(np.array([(0.5, 0.5, 0.5)]))
    # fragments = fragment_voxel_grid(grid, side, stride, roi_indices)
    # print(fragments.shape)

    # plt.imshow(fragments[4, 1, 4, :, :])
    # plt.show()
    