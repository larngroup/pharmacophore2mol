import os
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
import torch
from pharmacophore2mol.data.voxelizer import Voxelizer, get_frag_count, fragment_voxel_grid
from pharmacophore2mol.data.pharmacophore import Pharmacophore, PHARMACOPHORE_CHANNELS
from pharmacophore2mol.data.utils import get_translation_vector, translate_mol, mol_to_atom_dict
from config import config

class SubGridsDataset(Dataset):
    def __init__(self, mols_filename):
        self.mols_filename = mols_filename
        self.mol_supplier = Chem.SDMolSupplier(mols_filename, removeHs=False, sanitize=True, strictParsing=False)
        self.n_mols = len(self.mol_supplier)
        self.n_samples = 0
        self.index = []
        for i, mol in enumerate(self.mol_supplier):
            if mol is not None:
                count = self._count_samples_in_mol(mol)
                if count > 0:
                    self.n_samples += count
                    self.index.extend([i] * count)
    
    def _count_samples_in_mol(self, mol):
        translation = get_translation_vector(mol.GetConformer().GetPositions())
        mol = translate_mol(mol, translation)
        pharmacophore = Pharmacophore.from_mol(mol, ignore_directions=True)
        print(pharmacophore.to_dict())
        mol_v = Voxelizer(channels=[], resolution=config["resolution"], mode="dry_run")
        atom_dict = mol_to_atom_dict(mol)
        dummy_grid = mol_v.voxelize(atom_dict)
        side = mol_v.distance_to_voxel(config["side"])
        stride = mol_v.distance_to_voxel(config["stride"])
        roi_indices = mol_v.get_indexes(pharmacophore.get_coordinates())
        frag_count = get_frag_count(dummy_grid.shape[1:], roi_indices, side, stride)
        return frag_count


    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        mol_idx = self.index[idx]
        try:
            mol = self.mol_supplier[mol_idx]
        except IndexError:
            raise IndexError(f"Index {idx} out of range for molecule supplier.")
        if mol is None:
            raise ValueError(f"Invalid molecule at index {idx} in {self.mols_filename}.")
        
        translation = get_translation_vector(mol.GetConformer().GetPositions())
        mol = translate_mol(mol, translation)
        pharmacophore = Pharmacophore.from_mol(mol, ignore_directions=True)
        mol_v = Voxelizer(channels=config["channels"], resolution=config["resolution"], mode="gaussian")
        pharm_v = Voxelizer(channels=PHARMACOPHORE_CHANNELS, resolution=config["resolution"], mode="gaussian")
        atom_dict = mol_to_atom_dict(mol)
        pharm_dict = pharmacophore.to_dict()
        print(pharm_dict)
        print(atom_dict)
        atom_grid = mol_v.voxelize(atom_dict)
        side = mol_v.distance_to_voxel(config["side"])
        stride = mol_v.distance_to_voxel(config["stride"])
        roi_indices = mol_v.get_indexes(pharmacophore.get_coordinates())
        
        return None, None
    


if __name__ == "__main__":
    # Load a molecule
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    # suppl = Chem.SDMolSupplier("./raw/zinc3d_test.sdf", removeHs=False, sanitize=False, strictParsing=False)
    # mol = suppl[0]
    # # Extract pharmacophore features
    # print("\nExtracting Pharmacophore Features...")
    dataset = SubGridsDataset(mols_filename="../../data/raw/original_benzene.sdf")
    # dataset[0]