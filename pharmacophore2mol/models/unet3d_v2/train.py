from datetime import datetime
import random
from tqdm import tqdm
import torch
from pharmacophore2mol.models.unet3d_v2.model import UNet3DV2
from pharmacophore2mol.models.unet3d_v2.dataset import NoisySubGridsDataset
from pharmacophore2mol.data.utils import RandomRotateMolTransform, RandomFlipMolTransform
import os
from torch.utils.tensorboard import SummaryWriter


N_EPOCHS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"



def train(model, train_loader, optimizer, criterion, device, tb_logger=None):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", unit="batch")
    model.train()
    for batch_idx, (_, _, noised_mol_frag, added_noise, timestep) in loop:
        noised_mol_frag = noised_mol_frag.to(device)
        added_noise = added_noise.to(device)
        timestep = timestep.to(device)

        optimizer.zero_grad()
        # Forward pass
        output = model(noised_mol_frag, timestep)
        # Compute loss
        loss = criterion(output, added_noise)
        # Backward pass
        loss.backward()
        optimizer.step()
        # Update progress bar and logging
        loop.set_postfix(loss=loss.item())
        loop.refresh()
        if tb_logger:
            tb_logger.add_scalar("train/loss", loss.item(), batch_idx + epoch * len(train_loader))




def evaluate(model, val_loader, criterion, device):
    ...


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    #logging and viz
    writer = SummaryWriter(log_dir=f"runs/exp_at_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # Define the dataset and dataloader
    train_dataset = NoisySubGridsDataset(mols_filepath="../../data/raw/original_phenol.sdf", force_len=10000, transforms=[], return_clean=True)
    val_dataset = NoisySubGridsDataset(mols_filepath="../../data/raw/original_phenol.sdf", force_len=100, transforms=[], return_clean=True)
    example_batch = [train_dataset[i] for i in random.sample(range(len(train_dataset)), 16)]
    #plot example_batch distribution
    writer.add_histogram("example_batch/clean_mol_frag", torch.stack([x[0] for x in example_batch]), bins=1000, global_step=0)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, persistent_workers=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, persistent_workers=True)
    model = UNet3DV2(in_channels=3,
                    out_channels=3,
                    n_internal_channels=32,
                    ch_mults=[1, 2, 2, 4],
                    is_attn=[False, False, True, True],
                    n_blocks=2,
                    n_groups=1
                    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.MSELoss()
    for epoch in range(N_EPOCHS):
        train(model, train_dataloader, optimizer, loss, device, tb_logger=writer)

        if (epoch + 1) % 5 == 0:
            evaluate(model, val_dataloader, loss, device)

    writer.close()
        


            
        
        