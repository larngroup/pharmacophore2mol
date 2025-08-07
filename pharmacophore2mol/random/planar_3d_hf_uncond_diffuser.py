# as seen in https://huggingface.co/docs/diffusers/tutorials/basic_training

from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from local_diffusers import DDPMPipeline3D, DDPMScheduler #NOTE: the 3d model that they implement is actually 2D + 1D (temporal), so it is actually for video
from local_diffusers import UNet3DModel, UNet2DModel #local version of the model that supports 3D convolutions
from PIL import Image
import torch.nn.functional as F
from local_diffusers.optimization import get_cosine_schedule_with_warmup
from local_diffusers.utils import make_image_grid
import os
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import Dataset



from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
from pharmacophore2mol.data.utils import RandomFlipMolTransform, RandomRotateMolTransform
import pharmacophore2mol as p2m


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
                RandomFlipMolTransform(planes=(True, True, True)),
                RandomRotateMolTransform(angles=(359, 359, 359)),
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

os.chdir(os.path.join(os.path.dirname(__file__), "."))



@dataclass
class TrainingConfig:
    image_size = 32
    train_batch_size = 4
    eval_batch_size = 16
    num_epochs = 510
    gradient_accumulation_steps = 4
    learning_rate = 1e-4 #TODO: revert back to 1e-4 if needed
    lr_warmup_steps = 500
    save_image_epochs = 30
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = "./saves/ddpm-planar_3d_new_loss_cosine"
    overwrite_output_dir = True
    seed = 0
    push_to_hub = False


config = TrainingConfig()


# Load the dataset

dataset = UncondDataset()
# config.dataset_name = "huggan/smithsonian_butterflies_subset"
# dataset = load_dataset(config.dataset_name, split="train")



#visualize the dataset

# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["image"]):
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# plt.show()


# preprocess = transforms.Compose( #TODO: IS THIS NORMALIZE USEFUL?
#     [
#         transforms.Resize((config.image_size, config.image_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )

# def transform(examples):
#     # images = [preprocess(image.convert("RGB")) for image in examples["image"]]
#     images = [preprocess(image) for image in examples["image"]]
#     # images = [image for image in examples["image"]]
#     return {"images": images}


# dataset.set_transform(transform)


#visualize the dataset

# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["images"]):
#     image = image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# plt.show()



train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


# for i in train_dataloader:
#     print(i.shape)
#     exit()

model = UNet3DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=[128, 128, 256, 256, 512, 512],
    # block_out_channels=[64, 128, 128, 256, 256, 512],
    # block_out_channels=[64, 64, 128, 128, 256, 256],
    down_block_types=(
        "DownBlock3D",
        "DownBlock3D",
        "DownBlock3D",
        "DownBlock3D",
        "AttnDownBlock3D",
        "DownBlock3D",
    ),
    up_block_types=(
        "UpBlock3D",
        "AttnUpBlock3D",
        "UpBlock3D",
        "UpBlock3D",
        "UpBlock3D",
        "UpBlock3D",
    ),
)


sample_image = dataset[0].unsqueeze(0)  # Add batch dimension
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)
plt.imshow(sample_image[0, :, :, :, 0].cpu().numpy().transpose(1, 2, 0)) #watch out cuz it is clipping values to [0, 1] range
plt.title("Sample Image")
plt.axis("off")
plt.show()
# exit()

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

# Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8)[0].numpy()).show()



noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)



optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs) // config.gradient_accumulation_steps,  # Adjusted for gradient accumulation. there's a bug i believe: as the lr_scheduler.step() appears to also, be wrapped by acumulate(), the steps should be reduced accordingly, or else it will update n times slower, not allowing finetuning
)



def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),# Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    #Make agrid of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    #save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    print(f"Saving images to {test_dir}/{epoch:04d}.png")
    image_grid.save(f"{test_dir}/{epoch:04d}.png")



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")


        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )
                

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise, reduction="none")
                mask = (batch + 1) / 2
                loss = (loss * mask).mean()  # Apply the mask to the loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline3D(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)



from accelerate import notebook_launcher
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args=args, num_processes=1)

