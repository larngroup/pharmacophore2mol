"""
This module is intended to provide dataset classes that adapt all functionality from the data package to PyTorch's Dataset class.
It is structured as a collection of DatasetWrapper subclasses that are intended to be used as wrappers to add functionality as needed.
DatasetWrapper is a thin wrapper around torch.utils.data.Dataset that adds no functionality by itself.
"""


import logging
from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm
from collections.abc import Iterable

from pharmacophore2mol.data.utils import (
    CustomSDMolSupplier, 
    get_translation_vector, 
    translate_mol,
    mol_to_atom_dict
)
from pharmacophore2mol.data.voxelizer import Voxelizer, get_frag_count, fragment_voxel_grid
from pharmacophore2mol.data.pharmacophore import Pharmacophore, PHARMACOPHORE_CHANNELS

logger = logging.getLogger(__name__)


class SDFDataset(Dataset):
    """
    Base dataset for loading molecules from an SDF file.
    
    This is the minimal base dataset - it only loads molecules and handles indexing.
    No transformations, no processing, just raw RDKit Mol objects.
    
    Parameters
    ----------
    sdf_filepath : str
        Path to the SDF file.
    """
    def __init__(self, sdf_filepath):
        self.sdf_filepath = sdf_filepath
        self.mol_supplier = CustomSDMolSupplier(self.sdf_filepath)
        self.n_mols = len(self.mol_supplier)

    def __len__(self):
        return self.n_mols
    
    def __getitem__(self, idx):
        mol = self.mol_supplier[idx]
        if mol is None:
            raise ValueError(f"Invalid molecule at index {idx} in {self.sdf_filepath}.")
        if self.transform:
            mol = self.transform(mol)
        return mol

class DatasetWrapper(Dataset):
    """
    Base class for all dataset wrappers.
    
    Provides automatic delegation to the wrapped dataset, reducing boilerplate code.
    All wrappers should inherit from this class.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to wrap.
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]





    

