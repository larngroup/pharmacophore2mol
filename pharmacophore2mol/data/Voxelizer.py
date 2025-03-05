from numpy import ceil


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
    

    
    def voxelize(self, points: dict, box_size: tuple):
        """
        Voxelize a point cloud.
        points: dict
            A dictionary with the channel name as key and the points as value.
            The points should be a list of tuples, each tuple containing the coordinates of the point.
        box_size: tuple
            A tuple with the box dimensions. Dimentions should be passed as absolute values (meaning
            in the same units as the points), and not as the relative size according to the voxel resolution.
            The dimentions shall be interpreted as minimum values, and will be rounded up to the nearest
            multiple of the resolution.
            For example, if a box_size of (10, 10, 10) is passed with a resolution of 0.75, the actual
            box size will be rounded up to (10.5, 10.5, 10.5). The extra 0.5 padding will be added only to
            the maximum sides of the axes of the reference frame, meaning no centering will be made.
        """
        box_shape = tuple([int(ceil(size / self.resolution)) for size in box_size])
        print(box_shape)
        

if __name__ == "__main__":
    v = Voxelizer(channels=["C", "H"], resolution=0.75)
    print(v)
    print(v.get_channels())
    print(v.voxelize({"C": [(0, 0, 0), (1, 1, 1)], "H": [(0.5, 0.5, 0.5)]}, (1, 1, 1)))