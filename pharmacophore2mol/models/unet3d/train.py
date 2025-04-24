from dataset import SubGridsDataset
import os
from torch.utils.data import DataLoader




if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    # Define the dataset and dataloader
    dataset = SubGridsDataset(mols_filename="../../data/raw/zinc3d_test.sdf")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate through the dataloader
    for batch in dataloader:
        print("Batch shape:", batch.shape)
        # print("Batch data:", batch)