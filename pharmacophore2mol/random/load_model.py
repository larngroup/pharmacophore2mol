from local_diffusers import UNet3DModel
from safetensors.torch import load_file
import torch
import json
import os
import pharmacophore2mol as p2m
from local_diffusers import DDPMPipeline3D, DDPMScheduler

def load_unet_from_safetensors(config_path, safetensors_path, device='cpu'):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model from config
    unet = UNet3DModel.from_config(config)
    unet.to(device)
    

    # Load safetensors weights
    state_dict = load_file(safetensors_path, device=device)
    unet.load_state_dict(state_dict)

    return unet


def generate_samples(unet, num_samples=1, batch_size=1, device='cpu'):
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    pipeline = DDPMPipeline3D(unet, scheduler=scheduler)
    return pipeline(
        batch_size=16,
        generator=torch.Generator(device="cpu").manual_seed(0),# Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

# Example usage:
if __name__ == "__main__":
    config_path = p2m.PROJECT_DIR / "random" / "models" / "big_cosine_x5data_x8emb" / "config.json"
    safetensors_path = p2m.PROJECT_DIR / "random" / "models" / "big_cosine_x5data_x8emb" / "diffusion_pytorch_model_419.safetensors"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = load_unet_from_safetensors(config_path, safetensors_path, device)
    print("UNet model loaded successfully.")
    samples = generate_samples(unet, num_samples=1, batch_size=1, device=device)
    print(samples)