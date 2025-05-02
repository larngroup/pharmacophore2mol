import torch
from dataset import SubGridsDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet3d
from config import config



if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    # Define the dataset and dataloader
    dataset = SubGridsDataset(mols_filename="../../data/raw/zinc3d_test.sdf")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) #idk why but calls to __getitem__ from dataloader seem some ms slower than direct calls to __getitem__ from dataset, for the same indexes. even for slices. this is just about the __getitem__ call time, checked by profiling, and not about all other extra methods. TODO: investigate this further.
    model = UNet3d(in_channels=5, out_channels=8, features=[32, 64, 128, 256]).to(config["device"])
    for epoch in range(config["epochs"]):
        
        # Training loop
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for batch_idx, (data, targets) in loop:
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
            loop.set_postfix(loss=loss.item())    
    loop.close()
    print("Training complete.")

