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
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = UNet3d(in_channels=6, out_channels=3, features=[32, 64, 128, 256]).to(config["device"])
    for epoch in range(config["epochs"]):
        
        # Training loop
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for batch_idx, (data, targets) in loop:
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
            loop.set_description(f"Epoch {epoch + 1}/{config['epochs']}")
            loop.set_postfix(loss=loss.item())    
    loop.close()
    print("Training complete.")

