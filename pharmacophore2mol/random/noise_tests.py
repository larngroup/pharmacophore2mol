import numpy as np
import torch
from local_diffusers import DDPMScheduler
from local_diffusers.utils import make_image_grid
import pharmacophore2mol as p2m
from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import math

class UncondDataset(Dataset):
    """
    Does the same as SubGridsDataset, but just returns the first 2D Slice of a Phenol (cause it's planar)
    """

    def __init__(self):
        self.dataset = SubGridsDataset(
            mols_filepath=p2m.RAW_DATA_DIR / "small_planar.sdf",
            padding=0,
            transforms=[
                # RandomFlipMolTransform(planes=(True, True, False)),
                # RandomRotateMolTransform(angles=(0, 0, 359)),
                # RandomFlipMolTransform(planes=(True, True, True)),
                # RandomRotateMolTransform(angles=(359, 359, 359)),
            ],
            force_len=32 * 32 #it seems to be the size of the butterfly dataset, so lets use it for consistency
        )

    def __getitem__(self, idx):
        _, mol_frag = self.dataset[idx]


        mol_frag = transforms.Normalize(0.5, 0.5)(mol_frag)  # Normalize to [-1, 1] (essentially a rescale). if this is not done, images get foggy, maybe because activation function range is not being fully utilized (hypothetical)
        # mol_slice = mol_frag[:, :, :, 0].squeeze(-1) #C, H, W, D
        return mol_frag#.permute(2, 0, 1)  # Change from (C, H, W) to (H, W, C) for visualization purposes
    def __len__(self):
        return len(self.dataset)
    

def get_arcsin_betas(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Generate betas using an arcsin function.
    This is the distribution of values of a sin function.
    Found the shape interesting, sort of an inverse sigmoid, so decided to try it.
    Check "arcsin distribution" for more details.
    """
    betas = torch.linspace(0, 1, num_timesteps)
    betas = beta_start + (beta_end - beta_start) * (2 / math.pi) * torch.arcsin(torch.sqrt(betas))  # Apply arcsin to the betas
    return betas

dataset = UncondDataset()
image: torch.Tensor = dataset[0]  # Get the first image from the dataset
clean_images = torch.vstack([image.unsqueeze(0)] * 16)  # Stack the images into a batch
print(clean_images.shape)
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_start=0.0, beta_end=1.0, rescale_betas_zero_snr=False)
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.0, beta_end=1.0, rescale_betas_zero_snr=False)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_start=0.0001, beta_end=0.02, rescale_betas_zero_snr=True)
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", beta_start=0.0, beta_end=1.0, rescale_betas_zero_snr=False)

trained_betas = get_arcsin_betas(1000, beta_start=0.0001, beta_end=0.02)


# print(trained_betas)
print(noise_scheduler.betas)

noise = torch.randn_like(clean_images)  # Generate random noise
# timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device)
timesteps = torch.linspace(0, noise_scheduler.config.num_train_timesteps - 1, steps=clean_images.shape[0], dtype=torch.long, device=clean_images.device)
print(timesteps)
noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)


plt.rcParams['text.usetex'] = True
plt.figure()
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, trained_betas=trained_betas, rescale_betas_zero_snr=False)
plt.plot(list(range(1000)), noise_scheduler.alphas_cumprod, label="Arcsine")
for test, label in zip(["linear", "scaled_linear", "sigmoid", "squaredcos_cap_v2"], ["Linear", "Scaled Linear", "Sigmoid", "Cosine"]):
    testing = DDPMScheduler(num_train_timesteps=1000, beta_schedule=test, beta_start=0.0001, beta_end=0.02, rescale_betas_zero_snr=True)
    plt.plot(list(range(1000)), testing.alphas_cumprod, label=label)
plt.xlabel("Timestep $t$", fontdict={"fontsize": 14})
plt.ylabel(r"$\bar{\alpha}_t$", fontdict={"fontsize": 20})
plt.legend()
plt.show()
noisy_rescaled_images = ((noisy_images + 1) / 2).clamp(0, 1)  # Rescale to [0, 1] for visualization  
plotting_slices = noisy_rescaled_images[:, :, :, :, 0]
pil_images = [to_pil_image(img) for img in plotting_slices]
image_grid = make_image_grid(pil_images, rows=4, cols=4)

image_grid.show()  # Display the image grid