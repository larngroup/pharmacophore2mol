import numpy as np


class Voxelizer:
    """
    This class is intended to be used to voxelize a point cloud, like some molecular representations or pharmacophores.
    It goes with a fixed resiolution approach instead of a fixed grid size approach.
    This means that, without proper padding, the output will be variable sized voxel grids.
    """
    def __init__(self, channels, resolution, mode="binary"):
        self.channels = {c: i for i, c in enumerate(channels)}
        self.mode = mode
        self.resolution = resolution

    def __repr__(self):
        return f"Voxelizer(channels={self.channels}, resolution={self.resolution}, mode={self.mode})"
    
    def get_channels(self):
        return self.channels
    

    
    def voxelize(self, points: dict, min_grid_size: tuple | None = None, allow_negative_coords=False): #python 3.10+ stuff, not that portable but pytorch already kinda limits backport to earlier python
        """
        Voxelize a point cloud.

        Parameters
        ----------
        points: dict
            A dictionary with the channel name as key and the points as value.
            The points should be a 2d array, with shape (#points, 3)

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
        coords = coords / self.resolution #TODO: check if this is correct
        indexes = np.floor(coords).astype(int)
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
    



    def fragment(grid, side, stride=None, filter_empty=True, point_cloud=None):
        """
        Fragment a voxel grid into smaller grids, cubic, fixed size, possibly overlapping grids.
        It fragments the grid into smaller grids of size side x side x side, with a stride of stride.
        (Therefore, if stride < side, the grids will overlap, as it is intentional sometimes)
        If stride is None, it will default to side.
        """

        assert not (point_cloud is None and filter_empty), "If no point cloud is passed, filter_empty should be False, as no filtering can be performed"
        assert grid.ndim == 4, "Grid should be a 4d array (channels, x, y, z)"
        assert side > 0, "Side should be a positive integer"
        assert (stride > 0 or stride is None), "Stride should be a positive integer"
        assert grid.shape[1] >= side, "Grid side should be smaller than the grid size"
        assert grid.shape[2] >= side, "Grid side should be smaller than the grid size"
        assert grid.shape[3] >= side, "Grid side should be smaller than the grid size"

        if stride is None:
            stride = side

        fragments = []
        for i in range(0, grid.shape[1] - side + 1, stride):
            for j in range(0, grid.shape[2] - side + 1, stride):
                for k in range(0, grid.shape[3] - side + 1, stride):
                    fragment = grid[:, i:i+side, j:j+side, k:k+side]
                    if filter_empty and np.sum(fragment) == 0:
                        continue
                    fragments.append(fragment)







if __name__ == "__main__":
    v = Voxelizer(channels=["C", "H", "N"], resolution=0.05, mode="gaussian")
    grid = v.voxelize({"C": [(0, 0, 0), (1, 1, 1)], "H": [(0.5, 0.5, 0.5)]}, (1, 1, 1))
    
    import matplotlib.pyplot as plt
    plt.imshow(grid[0, 0, :, :])
    plt.show()
    