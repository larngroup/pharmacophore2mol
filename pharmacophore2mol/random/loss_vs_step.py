import pharmacophore2mol as p2m
from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
from pharmacophore2mol.data.utils import RandomFlipMolTransform, RandomRotateMolTransform

from torch.utils.data import Dataset
from torchvision import transforms



class UncondSliceDataset(Dataset):
    """
    Does the same as SubGridsDataset, but just returns the first 2D Slice of a Phenol (cause it's planar)
    """

    def __init__(self):
        self.dataset = SubGridsDataset(
            mols_filepath=p2m.RAW_DATA_DIR / "small_planar.sdf",
            padding=0,
            transforms=[
                RandomFlipMolTransform(planes=(True, True, False)),
                RandomRotateMolTransform(angles=(0, 0, 359)),
            ],
            force_len=32 * 32 #it seems to be the size of the butterfly dataset, so lets use it for consistency
        )


    def __getitem__(self, idx):
        _, mol_frag = self.dataset[idx]


        mol_frag = transforms.Normalize(0.5, 0.5)(mol_frag)  # Normalize to [-1, 1] (essentially a rescale). if this is not done, images get foggy, maybe because activation function range is not being fully utilized (hypothetical)
        mol_slice = mol_frag[:, :, :, 0].squeeze(-1) #C, H, W, D
        return mol_slice
    def __len__(self):
        return len(self.dataset)
    




# PLANNAR_CHECKPOINT_2D = 

# dataset = UncondSliceDataset()
# model = 