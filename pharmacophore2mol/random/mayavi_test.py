from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


values = np.random.rand(10, 10, 10, 3)  # Example values, shape (X, Y, Z, 32)


# Create RGBA facecolors based on values
# You can customize the color part as well
facecolors = np.zeros((*values.shape[:3], 4), dtype=np.float32)  # (X, Y, Z, 4)
print(facecolors.shape)

transparency = np.mean(values, axis=-1)  # Example: mean value for transparency
print(transparency.shape)

# Set RGB to some color (e.g. blue), alpha to value
facecolors[..., :3] = values
facecolors[..., 3] = transparency  # Alpha (transparency based on value)




print(values[:,:,:, 1].shape)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(values[:,:,:, 0], facecolors=facecolors, edgecolor=None)

plt.show()