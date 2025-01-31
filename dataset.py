from datetime import datetime
import glob
import os
import re
# import torch
import pytorch_lightning as pl
from xxhash import xxh64
from pathlib import Path

class PharmacophoreVoxelDataModule(pl.LightningDataModule):
    def __init__(self, data_file, data_dir="./data", batch_size=32, num_workers=4):
        # super.__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.data_file = self.data_dir / data_file
        #check file exists
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File not found: {data_file}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        pass

    def prepare_data(self):
        #check if cached
        cache_filename, cache_file_exists = _get_dataset_cache(self.data_file, self.cache_dir) #no state assignment here!!! (no self.x = y)
        if not cache_file_exists:
            with open(self.cache_dir / cache_filename, "wb") as f:
                #save the processed dataset
                f.write(f"Hello World at {datetime.now()}".encode())
                
                
            

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass



def _get_dataset_cache(data_file: Path, cache_dir: Path):
    """
    Returns True if the dataset is cached, False otherwise.
    A dataset cache will be named as follows:
    if the data_file is "data.sdf", the cache will be "data.sdf.<hash>.cache"
    Also, it deletes previous cached versions of the same data_file.
    """
    #calc hash of data_file
    hash = _get_dataset_hash(data_file)
    cache_filename = data_file.name + "." + hash + ".cache"
    possible_files = glob.glob(data_file.name + ".*.cache", root_dir=cache_dir)
    exists = False
    if len(possible_files) == 0:
        return cache_filename, exists
    
    for f in possible_files:
        if f == cache_filename:
            exists = True
        else:
            os.remove(cache_dir / f)
    return cache_filename, exists

def _get_dataset_hash(data_file: Path):
    hash = xxh64()
    with open(data_file, "rb") as f:
        while chunk := f.read(1024 * 1024 * 100):  # Read file in 100MB chunks
            hash.update(chunk)
    return hash.hexdigest()
    




if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    data = PharmacophoreVoxelDataModule("zinc3d_test.sdf")
    print(data.prepare_data())