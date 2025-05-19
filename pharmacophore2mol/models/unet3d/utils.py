from matplotlib import pyplot as plt
import numpy as np
from pharmacophore2mol.data.utils import plot_voxel_grid_sweep
import matplotlib.animation as animation

def get_next_multiple_of(x: int, y: int) -> int: #this is useful for padding the input to the model, so Pool Layers can be used without issues
    """
    Get the next multiple of x that is greater than or equal to y.
    """
    if x <= 0 or y < 0:
        raise ValueError("x must be greater than 0 and y must be non-negative.")
    return ((y + x - 1) // x) * x


def save_preds_as_gif(trues: np.ndarray, preds: np.ndarray, channel: int, filename: str, n_preds: int = 5):
    """
    Save an animation of random noise as a GIF.
    The resulting GIF will have 2 rows and n_preds columns.
    Each subplot will display an animation of random noise.
    """
    fig, axs = plt.subplots(2, n_preds, figsize=(n_preds * 4, 8))
    fig.suptitle("True Grids (UP) vs Predictions (DOWN)", fontsize=16)

    preds_to_plot_idxs = np.random.choice(preds.shape[0], n_preds, replace=False)  # Randomly select n_preds indices
    preds_to_plot = preds[preds_to_plot_idxs, :, :, :]
    trues_to_plot = trues[preds_to_plot_idxs, :, :, :]

    def update(frame):
        for i in range(2):  # Two rows
            set_to_plot = trues_to_plot if i == 0 else preds_to_plot
            for j in range(n_preds):  # n_preds columns
                ax = axs[i, j]
                ax.clear()
                ax.axis('off')  # Turn off axis for better visualization
                # noise = np.random.rand(preds.shape[2], preds.shape[3])  # Generate random noise
                ax.imshow(set_to_plot[j, channel, :, :, frame], cmap='viridis', animated=True, vmin=0, vmax=1)

    ani = animation.FuncAnimation(fig, update, frames=list(range(preds.shape[4])) + list(range(preds.shape[4] - 2, -1, -1)), interval=50)
    ani.save(filename, writer='imagemagick', fps=10)
        



if __name__ == "__main__":
    # Example usage
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    preds = np.random.rand(20, 5, 5, 5, 19)  # Example predictions
    save_preds_as_gif(preds, preds, channel=0, filename="./saves/predictions.gif", n_preds=3)