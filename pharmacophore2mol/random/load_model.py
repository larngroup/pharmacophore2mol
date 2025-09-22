import PIL
from local_diffusers import UNet3DModel, UNet2DModel
from safetensors.torch import load_file
import torch
import json
import os
import pharmacophore2mol as p2m
from local_diffusers import DDPMPipeline3D, DDPMScheduler, DDPMPipeline
from local_diffusers.utils import make_image_grid
from datetime import datetime




def load_unet_from_safetensors(config_path, safetensors_path, device='cpu'):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model from config
    unet = UNet3DModel.from_config(config)
    unet.eval()
    unet.to(device)
    

    # Load safetensors weights
    state_dict = load_file(safetensors_path, device=device)
    unet.load_state_dict(state_dict)

    return unet

def load_unet2d_from_safetensors(config_path, safetensors_path, device='cpu'):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model from config
    unet = UNet2DModel.from_config(config)
    unet.eval()
    unet.to(device)
    # Load safetensors weights
    state_dict = load_file(safetensors_path, device=device)
    unet.load_state_dict(state_dict)

    return unet

def generate_samples(unet, batch_size=16, device='cpu'):
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    pipeline = DDPMPipeline3D(unet, scheduler=scheduler)
    return pipeline(
        batch_size=batch_size,
        num_inference_steps=1000,
        generator=torch.Generator(device="cpu").manual_seed(1),# Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

def generate_samples_2d(unet, batch_size=16, device='cpu'):
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = DDPMPipeline(unet, scheduler=scheduler)
    samples = pipeline(
        batch_size=batch_size,
        num_inference_steps=1000,
        generator=torch.Generator(device="cpu").manual_seed(1),# Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    return samples
   

# Example usage:
if __name__ == "__main__":
    # model_name = "ddpm-plannar_3d_x8_x5data_cosine"
    model_name = "ddpm-planar_aug_norm-32"
    config_path = p2m.PROJECT_DIR / "random" / "saves" / model_name / "unet" / "config.json"
    safetensors_path = p2m.PROJECT_DIR / "random" / "saves" / model_name / "unet" / "diffusion_pytorch_model.safetensors"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # unet = load_unet_from_safetensors(config_path, safetensors_path, device)
    unet = load_unet2d_from_safetensors(config_path, safetensors_path, device)
    unet.config.sample_size = 128
    #change configs of image size
    print("UNet model loaded successfully.")
    timestamp = round(10000000000 - datetime.now().timestamp())
    out_dir = os.path.join(p2m.PROJECT_DIR / "random" / "saves" / "samples")
    os.makedirs(out_dir, exist_ok=True)

    # samples = generate_samples(unet, batch_size=16, device=device)
    samples = generate_samples_2d(unet, batch_size=1, device=device)
    image_grid = make_image_grid(samples, rows=1, cols=1)

    
    print(f"Saving images to {out_dir}/{timestamp}.png")
    image_grid.save(f"{out_dir}/{timestamp}.png")