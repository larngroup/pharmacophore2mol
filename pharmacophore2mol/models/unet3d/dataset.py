import os
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
import torch
from tqdm import tqdm
from pharmacophore2mol.data.voxelizer import Voxelizer, get_frag_count, fragment_voxel_grid
from pharmacophore2mol.data.pharmacophore import Pharmacophore, PHARMACOPHORE_CHANNELS
from pharmacophore2mol.data.utils import get_translation_vector, translate_mol, mol_to_atom_dict
from pharmacophore2mol.models.unet3d.config import config
import bisect

class SubGridsDataset(Dataset):
    def __init__(self, mols_filepath):
        self.mols_filepath = mols_filepath
        self.mol_supplier = self._get_mol_supplier()
        self.n_mols = len(self.mol_supplier)
        self.n_samples = 0
        # self.index = [] #large structures like these cause workers to take quite long to launch, especially on windows, where there's a spawn() followed by pickling (ForkPickler.dump() seems to jump from milisseconds to 3 secods whenever I have an index larger than 3k elements). Use persistent workers to address this. in linux should not be a problem, but need to test
        self.cumsum_index = [] #TODO: are both indexes really needed for performance? test this
        for i, mol in tqdm(enumerate(self.mol_supplier), total=self.n_mols, desc="Indexing dataset", unit="mol"):
            if mol is not None:
                count = self._count_samples_in_mol(mol)
                # if count > 0: #commented cause we want to include empty molecules for debugging and preventing malformed dataset crashing
                self.cumsum_index.append(self.n_samples) #WARNING: cumulative sum BEFORE the current molecule, not the current one. this is important for indexing
                self.n_samples += count
                # self.index.extend([i] * count)

        # print(self.cumsum_index)
    
    def _count_samples_in_mol(self, mol):
        translation = get_translation_vector(mol.GetConformer().GetPositions())
        mol = translate_mol(mol, translation)
        pharmacophore = Pharmacophore.from_mol(mol, ignore_directions=True)
        mol_v = Voxelizer(channels=[], resolution=config["resolution"], mode="dry_run")
        side = mol_v.distance_to_voxel(config["side"])
        stride = mol_v.distance_to_voxel(config["stride"])
        atom_dict = mol_to_atom_dict(mol)
        dummy_grid = mol_v.voxelize(atom_dict, min_grid_size=(config["side"], config["side"], config["side"]))
        roi_indices = mol_v.get_indexes(pharmacophore.get_coordinates())
        frag_count = get_frag_count(dummy_grid.shape[1:], side, stride, roi_indices)
        return frag_count


    def __len__(self):
        return self.n_samples
    
    # def _get_mol_idx(self, idx):
    #     return self.index[idx]
    
    def _get_mol_idx(self, idx): #binary search to be efficient with a cumsum index while sparing the memory of a full index (see worker's init delay issues on windows)
        return bisect.bisect_right(self.cumsum_index, idx) - 1
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idxs = range(*idx.indices(self.n_samples))
            return [self[i] for i in idxs]
        elif isinstance(idx, int):
            if idx < -self.n_samples or idx >= self.n_samples:
                raise IndexError(f"Index {idx} out of range for dataset of size {self.n_samples}.")
            if idx < 0:
                idx += self.n_samples
            mol_idx = self._get_mol_idx(idx) #TODO: while this is O(1), we could look into using the cumsum_index for memory reasons, especially in a multiprocess environment like pytorch multiple workers (indexes are copied)
            frag_idx = idx - self.cumsum_index[mol_idx]
            try:
                mol = self.mol_supplier[mol_idx]
            except IndexError:
                raise IndexError(f"Index {idx} out of range for molecule supplier.")
            if mol is None:
                raise ValueError(f"Invalid molecule at index {idx} in {self.mols_filepath}.")
            
            translation = get_translation_vector(mol.GetConformer().GetPositions())
            mol = translate_mol(mol, translation)
            pharmacophore = Pharmacophore.from_mol(mol, ignore_directions=True)
            mol_v = Voxelizer(channels=config["channels"], resolution=config["resolution"], mode="gaussian")
            pharm_v = Voxelizer(channels=PHARMACOPHORE_CHANNELS, resolution=config["resolution"], mode="gaussian")
            atom_dict = mol_to_atom_dict(mol)
            pharm_dict = pharmacophore.to_dict()
            atom_grid = mol_v.voxelize(atom_dict, min_grid_size=(config["side"], config["side"], config["side"]))
            pharm_grid = pharm_v.voxelize(pharm_dict, force_shape=atom_grid.shape[1:])

            #all till here could be saved or cached in the future
            
            side = mol_v.distance_to_voxel(config["side"])
            stride = mol_v.distance_to_voxel(config["stride"])
            roi_indices = mol_v.get_indexes(pharmacophore.get_coordinates())
            mol_frags = fragment_voxel_grid(atom_grid, side, stride, roi_indices)
            pharm_frags = fragment_voxel_grid(pharm_grid, side, stride, roi_indices) #TODO: optimize this. the index calculations should only be done once
            mol_frag = mol_frags[frag_idx]
            pharm_frag = pharm_frags[frag_idx]
            return mol_frag, pharm_frag
    
    def _get_mol_supplier(self):
        return Chem.SDMolSupplier(self.mols_filepath, removeHs=False, sanitize=True, strictParsing=False)


    def __getstate__(self):
        #file handles are not serializable, got to remove them to work on windows
        #i believe this is unnecessary on linux as dataloader uses fork() instead of spawn() by default, and that seems to support copying the file handles between processes
        #anyway, this is more portable like this
        state = self.__dict__.copy() #is copy enough here? or do we need to deepcopy?
        # del state["mol_supplier"] #this or setting it to None?
        state["mol_supplier"] = None #this is probably not necessary, but just in case]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mol_supplier = self._get_mol_supplier()
        


if __name__ == "__main__":
    # Load a molecule
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    # suppl = Chem.SDMolSupplier("./raw/zinc3d_test.sdf", removeHs=False, sanitize=False, strictParsing=False)
    # mol = suppl[0]
    # # Extract pharmacophore features
    # print("\nExtracting Pharmacophore Features...")
    # dataset = SubGridsDataset(mols_filename="../../data/raw/original_benzene.sdf")
    dataset = SubGridsDataset(mols_filepath="../../data/raw/zinc3d_test.sdf")
    
    tensor_x, tensor_y = dataset[16]
    # batch = dataset[:16]
    # print(tensor_x.shape, tensor_y.shape)
