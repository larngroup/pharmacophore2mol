from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset


class Custom3ChannelDataset(SubGridsDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return y #not very performant but wtv
    

