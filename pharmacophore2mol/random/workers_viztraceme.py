from time import sleep
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Simulate some work
        # sleep(0.4)
        return torch.randn(10), torch.randn(10)  # Simulate a target as well


if __name__ == "__main__":
    dataset = MyDataset(1000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True, desc="Loading data", unit="batch")
    for batch in loop:
        pass