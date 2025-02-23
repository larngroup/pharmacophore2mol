import numpy as np
import matplotlib.pyplot as plt

# Create a 3D boolean array (2x2x2 grid of subunit voxels)
filled = np.ones((2, 2, 2), dtype=bool)  # All voxels are filled

# Define voxel coordinates with a smaller spacing (0.5 instead of 1)
x, y, z = np.meshgrid(np.arange(3) * 0.5, np.arange(3) * 0.5, np.arange(3) * 0.5, indexing="ij")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot voxels
ax.voxels(x, y, z, filled, facecolors='cyan', edgecolor='k', alpha=0.7)

# Set limits and labels
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
