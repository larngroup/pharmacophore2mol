from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
from pharmacophore2mol.models.unet3d.dataset import SubGridsDataset
from pharmacophore2mol.models.unet3d.utils import save_preds_as_gif
from pharmacophore2mol.data.utils import RandomRotateMolTransform, RandomFlipMolTransform
from local_diffusers.utils import make_image_grid
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from rdkit import Chem
from pharmacophore2mol.data.voxelizer import Voxelizer
from pharmacophore2mol.data.utils import get_translation_vector, translate_mol, mol_to_atom_dict


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    transforms = []
    mol_supplier = Chem.SDMolSupplier("../data/raw/phenol_pharm_simple.sdf")
    pharm = mol_supplier[0]
    arom_center = pharm.GetConformer().GetPositions()[0]
    center = np.array([4, 4, 4])
    translate_vector = center - arom_center
    pharm = translate_mol(pharm, translate_vector)
    print(pharm.GetConformer(0).GetPositions())

    vox = Voxelizer(channels=["C", "H", "O"], resolution=0.25) #reduced res to fit in the box for viz
    # pharm = translate_mol(pharm, get_translation_vector(pharm.GetConformer().GetPositions(), padding=2.5))
    centered_pharms = []
    rot_class = RandomRotateMolTransform(angles=(0, 0, 359)) #just using cause of the private methods, but manually overriding __call__()
    for a in np.linspace(0, 2*np.pi, 17)[:-1]:
        _pharm = deepcopy(pharm)
        rotation_matrix = rot_class._get_rotation_matrix((0, 0, a))
        coords = _pharm.GetConformer(0).GetPositions()
        new_coords = (coords - center) @ rotation_matrix + center
        _pharm.GetConformer(0).SetPositions(new_coords)
        
        centered_pharms.append(_pharm)

    pharms_dict = [mol_to_atom_dict(p) for p in centered_pharms]
    pharms_voxel = [vox.voxelize(p, force_shape=(32, 32, 32)) for p in pharms_dict]
    pharms_voxel = Tensor(pharms_voxel)
    print(pharms_voxel.shape)


    plotting_slices = pharms_voxel[:, :, :, :, 16]
    pil_images = [to_pil_image(img) for img in plotting_slices]
    image_grid = make_image_grid(pil_images, rows=4, cols=4)

    image_grid.show()  # Display the image grid cant save for sm reason

    # plotting_slices = mols[:, :, :, :, 0]
    # pil_images = [to_pil_image(img) for img in plotting_slices]
    # image_grid = make_image_grid(pil_images, rows=4, cols=4)

    # image_grid.show()  # Display the image grid