import os
from torch.utils.data import Dataset
import numpy as np


class SubGridsDataset(Dataset):
    def __init__(self, mols_filename):
        self.mols_filename = mols_filename
        #preprocess the mols in mols_filename to get the subgrids prefetched

        

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace('.jpg', '_mask.gif'))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0  # Normalize to 1, there are only 2 values (0 and 255)


        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return image, mask