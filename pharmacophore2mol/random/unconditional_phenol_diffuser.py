import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pharmacophore2mol.models.unet3d.model import UNet3d
from pharmacophore2mol.models.unet3d.config import config

batch_size = 4
# in_channels = 6
out_channels = len(config["channels"])  # Assuming config["channels"] is a list of channel names
voxel_dim = 32  # 16x16x16 grid

# Example target data
x_start = torch.randn(batch_size, out_channels, voxel_dim, voxel_dim, voxel_dim)


timesteps = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timesteps)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


def q_sample(x_start, t, noise=None):
    """
    Sample from q(x_t | x_0) by adding noise.
    x_start: (B, C, X, Y, Z)
    t: timestep (B,)
    noise: if provided, uses this noise instead of sampling.
    """
    device = x_start.device

    if noise is None:
        noise = torch.randn_like(x_start)

    
    alphas_cumprod_device = alphas_cumprod.to(device)

    # Now indexing will work
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod_device[t]).view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod_device[t]).view(-1, 1, 1, 1, 1)

    return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise



# class Simple3DUNet(nn.Module):
#     def __init__(self, in_channels=3, base_channels=32):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_channels, base_channels, 3, padding=1)
#         self.conv2 = nn.Conv3d(base_channels, base_channels, 3, padding=1)
#         self.conv3 = nn.Conv3d(base_channels, in_channels, 3, padding=1)

#     def forward(self, x, t_emb):
#         # Here t_emb can be used for timestep embedding; for now, we ignore it for simplicity.
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.conv3(x)
#         return x



def training_step(model, x_start):
    batch_size = x_start.size(0)
    t = torch.randint(0, timesteps, (batch_size,), device=x_start.device).long()
    noise = torch.randn_like(x_start)
    
    # Generate noisy voxel grid
    x_noisy = q_sample(x_start, t, noise=noise)
    
    # Predict noise
    noise_pred = model(x_noisy, t)

    # MSE loss between predicted noise and true noise
    loss = F.mse_loss(noise_pred, noise)
    return loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet3d(in_channels=out_channels, out_channels=out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

x_start = x_start.to(device)

for epoch in range(10):
    loss = training_step(model, x_start)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")




@torch.no_grad()
def p_sample(model, x, t):
    beta_t = betas[t]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])
    sqrt_recip_alphas_t = torch.sqrt(1 / alphas[t])

    noise_pred = model(x, torch.tensor([t]*x.size(0), device=x.device))
    
    # Compute mean
    mean = sqrt_recip_alphas_t * (x - beta_t / sqrt_one_minus_alphas_cumprod_t * noise_pred)
    
    if t > 0:
        noise = torch.randn_like(x)
        sigma_t = torch.sqrt(beta_t)
        return mean + sigma_t * noise
    else:
        return mean

@torch.no_grad()
def sample_voxel_grid(model, shape):
    x = torch.randn(shape, device=next(model.parameters()).device)
    for t in reversed(range(timesteps)):
        x = p_sample(model, x, t)
    return x

# Example usage:
generated_voxel_grid = sample_voxel_grid(model, (1, out_channels, voxel_dim, voxel_dim, voxel_dim))
print("Generated voxel grid shape:", generated_voxel_grid.shape)
