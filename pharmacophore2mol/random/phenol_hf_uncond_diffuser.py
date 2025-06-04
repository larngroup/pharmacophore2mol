# as seen in https://huggingface.co/docs/diffusers/tutorials/basic_training

from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel #a 3d model seems to exist, but i got to check wether there's any resize happening in the residual connections
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from accelerate import notebook_launcher
from torchvision.transforms.functional import to_pil_image


os.chdir(os.path.join(os.path.dirname(__file__), "."))



@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 8
    eval_batch_size = 8
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = "./saves/ddpm-butterflies-128"
    overwrite_output_dir = True
    seed = 0
    push_to_hub = False


config = TrainingConfig()


# Load the dataset

config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")



#visualize the dataset

# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["image"]):
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# plt.show()


preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)


#visualize the dataset

# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["images"]):
#     image = image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# plt.show()



train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=[128, 128, 256, 256, 512, 512],
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)


sample_image = dataset[0]["images"].unsqueeze(0)  # Add batch dimension
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
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
    num_training_steps=len(train_dataloader) * config.num_epochs,
)



def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),# Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    #Make agrid of the images
    image_grid = make_image_grid(images, rows=2, cols=4)

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

        print_stats = True
        for step, batch in enumerate(train_dataloader):
            if print_stats and  (step / len(train_dataloader)) > 0.1:
                print_stats = False
                print(step)
                print(batch["images"][0][0][63])
                # Convert batch["images"] to a list of PIL RGB images
                def denormalize(tensor):
                    return (tensor * 0.5 + 0.5).clamp(0, 1)
                pil_images = [to_pil_image(denormalize(image), mode="RGB") for image in batch["images"]]
                image_grid = make_image_grid(pil_images, rows=2, cols=4)
                statsdir = os.path.join(config.output_dir, "stats")
                image_grid.save(f"{statsdir}/train_batch_{epoch}_{step}.png")  
            clean_images = batch["images"]
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
                loss = F.mse_loss(noise_pred, noise)
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
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

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




args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args=args, num_processes=1)

