from matplotlib import pyplot as plt
import numpy as np
from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
from pharmacophore2mol.models.unet3d.utils import save_preds_as_gif
from pharmacophore2mol.data.utils import RandomRotateMolTransform, RandomFlipMolTransform


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    dataset = SubGridsDataset(mols_filepath="../data/raw/original_phenol.sdf")
    transforms = [RandomFlipMolTransform(), RandomRotateMolTransform((0, 0, 360))] 
    dataset_t = SubGridsDataset(mols_filepath="../data/raw/original_phenol.sdf", transforms=transforms)

    up = np.expand_dims(dataset[0][0], axis=0)
    down = np.expand_dims(dataset[0][1], axis=0)
    up = up + np.roll(up, shift=1, axis=1) + np.roll(up, shift=2, axis=1)
    down = down# + np.roll(down, shift=1, axis=1) + np.roll(down, shift=2, axis=1)

    plt.imshow(down[0, 0, :, :, 0], cmap='viridis', animated=True, vmin=0, vmax=1)
    plt.show()
    plt.imshow(up[0, 0, :, :, 0], cmap='viridis', animated=True, vmin=0, vmax=1)
    plt.show()

    
    save_preds_as_gif(up, down, channel=0, filename="./saves/test_rotations.gif", n_preds=1)

    print("no transforms:", len(dataset))
    print("w/ transforms:", len(dataset_t))