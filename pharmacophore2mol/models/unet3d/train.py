import numpy as np
import torch
from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from pharmacophore2mol.models.unet3d.model import UNet3d
from pharmacophore2mol.models.unet3d.config import config
from pharmacophore2mol.models.unet3d.utils import save_preds_as_gif


if __name__ == "__main__":

    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    # Define the dataset and dataloader
    train_dataset = SubGridsDataset(mols_filepath="../../data/raw/original_phenol.sdf", force_len=config["batch_size"] * 100)
    val_dataset = SubGridsDataset(mols_filepath="../../data/raw/original_phenol.sdf", force_len=config["batch_size"] * 5)

    # dataset = SubGridsDataset(mols_filepath="../../data/raw/zinc3d_test.sdf")

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8, persistent_workers=True)#idk why but calls to __getitem__ from dataloader seem some ms slower than direct calls to __getitem__ from dataset, for the same indexes. even for slices. this is just about the __getitem__ call time, checked by profiling, and not about all other extra methods. TODO: investigate this further.
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8, persistent_workers=True)

    model = UNet3d(in_channels=8, out_channels=5, features=[32, 64, 128, 256]).to(config["device"])
    for epoch in range(config["epochs"]):
        # Training loop
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"Epoch {epoch + 1}/{config['epochs']}", unit="batch")
        
        for batch_idx, (data, targets) in train_loop:
            # exit()
            # pass
            data = data.to(config["device"])
            targets = targets.to(config["device"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            optimizer.zero_grad()
            predictions = model(data)
            with torch.amp.autocast(device_type=config["device"]):
                predictions = model(data)
                loss = torch.nn.functional.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()

            # Update the progress bar
            train_loop.set_postfix(loss=loss.item())
            train_loop.refresh()

        # end of epoch
        # Validation loop
        val_loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=True, desc=f"Validation {epoch + 1}/{config['epochs']}", unit="batch")
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, (data, targets) in val_loop:
                data = data.to(config["device"])
                targets = targets.to(config["device"])
                predictions = model(data)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                loss = torch.nn.functional.mse_loss(predictions, targets)

                # Update the progress bar
                val_loop.set_postfix(val_loss=loss.item())
                val_loop.refresh()
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        print("Saving predictions as GIF...")
        save_preds_as_gif(all_targets, all_preds, channel=0, filename=f"./saves/epoch_{epoch + 1}.gif", n_preds=5)

            

    # dataloader._get_iterator()._shutdown_workers()
    print("Training complete.")

