from torch.utils.data import Dataset

import os
import logging
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm
from pharmacophore2mol.data.voxelizer import Voxelizer, get_frag_count, fragment_voxel_grid
from pharmacophore2mol.data.pharmacophore import Pharmacophore, PHARMACOPHORE_CHANNELS
from pharmacophore2mol.data.utils import CustomSDMolSupplier, get_translation_vector, translate_mol, mol_to_atom_dict
from pharmacophore2mol.models.unet3d.config import config
from pharmacophore2mol.models.unet3d.utils import get_next_multiple_of
import bisect
from collections.abc import Iterable

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    def __init__(self, mols_filepath, padding=0, transforms=None, force_len=None):
        """
        Dataset for 3D pharmacophore and molecule voxel grid fragments.

        Parameters
        ----------
        mols_filepath : str
            Path to the molecule file (SDF format).
        padding : int, optional
            This is the minimum distance (in Angstroms) between the closest atom and the edge of the grid. Default is 0.
        transforms : callable, optional
            Optional transform or set of transforms to be applied to a sample.
        force_len : int, optional
            Force the dataset to have a specific length. Default is None.
            If provided, it will either truncate or repeat the original dataset to match this length.
        """
        self.mols_filepath = mols_filepath
        self.padding = padding
        self.transforms = transforms
        self.len = force_len
        self.mol_supplier = CustomSDMolSupplier(self.mols_filepath)
        self.n_mols = len(self.mol_supplier)
        self.n_samples = 0
        # self.index = [] #large structures like these cause workers to take quite long to launch, especially on windows, where there's a spawn() followed by pickling (ForkPickler.dump() seems to jump from milisseconds to 3 secods whenever I have an index larger than 3k elements). Use persistent workers to address this. in linux should not be a problem, but need to test
        self.cumsum_index = [] #TODO: are both indexes really needed for performance? test this
        for i, mol in (pbar := tqdm(enumerate(self.mol_supplier), total=self.n_mols, desc="Indexing dataset", unit="mol")):
            if mol is not None and (force_len is None or self.n_samples < force_len):
                count = self._count_samples_in_mol(mol)
                # if count > 0: #commented cause we want to include empty molecules for debugging and preventing malformed dataset crashing
                self.cumsum_index.append(self.n_samples) #WARNING: cumulative sum BEFORE the current molecule, not the current one. this is important for indexing
                self.n_samples += count
                # self.index.extend([i] * count)
            elif mol is not None: #interrupted due to force_len
                pbar.close()
                logger.info(f"Indexing interrupted due to force_len: Dataset length truncated to {force_len} samples, stopping indexing.")
                break
            else:
                logger.warning(f"Invalid molecule at index {i} in {self.mols_filepath}.")
        # print(self.cumsum_index)

        if force_len is not None:
            self.len = force_len
        else:
            self.len = self.n_samples
            
    def _load_and_transform(self, mol):
        if self.transforms is not None:
            if not isinstance(self.transforms, Iterable):
                self.transforms = [self.transforms]
            for transform in self.transforms:
                mol = transform(mol)
        rotated_coords = mol.GetConformer().GetPositions()
        translation = get_translation_vector(mol.GetConformer().GetPositions(), padding=self.padding)
        mol = translate_mol(mol, translation)
        return mol
    
    def _count_samples_in_mol(self, mol):
        mol = self._load_and_transform(mol)
        pharmacophore = Pharmacophore.from_mol(mol, ignore_directions=True)
        mol_v = Voxelizer(channels=[], resolution=config["resolution"], mode="dry_run")
        side = get_next_multiple_of(16, mol_v.distance_to_voxel(config["side"]))
        stride = mol_v.distance_to_voxel(config["stride"])
        atom_dict = mol_to_atom_dict(mol)
        dummy_grid = mol_v.voxelize(atom_dict, min_grid_shape=(side, side, side))
        roi_indices = mol_v.get_indexes(pharmacophore.get_coordinates())
        frag_count = get_frag_count(dummy_grid.shape[1:], side, stride, roi_indices)
        return frag_count


    def __len__(self):
        return self.len
    
    # def _get_mol_idx(self, idx):
    #     return self.index[idx]
    
    def _get_mol_idx(self, idx): #binary search to be efficient with a cumsum index while sparing the memory of a full index (see worker's init delay issues on windows)
        return bisect.bisect_right(self.cumsum_index, idx) - 1
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idxs = range(*idx.indices(self.len))
            return [self[i] for i in idxs]
        elif isinstance(idx, int):
            if idx < -self.len or idx >= self.len:
                raise IndexError(f"Index {idx} out of range for dataset of size {self.len}.")
            if idx < 0:
                idx += self.len

            #correct the idx for situations where dataset is repeated when force_len is set
            idx = idx % self.n_samples

            mol_idx = self._get_mol_idx(idx)
            frag_idx = idx - self.cumsum_index[mol_idx]
            # print("counts:", frag_idx, self._count_samples_in_mol(self.mol_supplier[mol_idx]), mol_idx, self.cumsum_index[mol_idx + 1] - self.cumsum_index[mol_idx])
            try:
                mol = self.mol_supplier[mol_idx]
            except IndexError:
                raise IndexError(f"Index {idx} out of range for molecule supplier.")
            if mol is None:
                raise ValueError(f"Invalid molecule at index {idx} in {self.mols_filepath}.")
            
            mol = self._load_and_transform(mol)
            pharmacophore = Pharmacophore.from_mol(mol, ignore_directions=True)
            mol_v = Voxelizer(channels=config["channels"], resolution=config["resolution"], mode=config["mode"], std=config["std"])
            pharm_v = Voxelizer(channels=PHARMACOPHORE_CHANNELS, resolution=config["resolution"], mode=config["mode"], std=config["std"])
            atom_dict = mol_to_atom_dict(mol)
            pharm_dict = pharmacophore.to_dict()
            side = get_next_multiple_of(16, mol_v.distance_to_voxel(config["side"]))
            stride = mol_v.distance_to_voxel(config["stride"])
            atom_grid = mol_v.voxelize(atom_dict, min_grid_shape=(side, side, side))
            pharm_grid = pharm_v.voxelize(pharm_dict, force_shape=atom_grid.shape[1:])

            #all till here could be saved or cached in the future
            
            
            roi_indices = mol_v.get_indexes(pharmacophore.get_coordinates())
            mol_frags = fragment_voxel_grid(atom_grid, side, stride, roi_indices)
            pharm_frags = fragment_voxel_grid(pharm_grid, side, stride, roi_indices) #TODO: optimize this. the index calculations should only be done once
            mol_frag = mol_frags[frag_idx]
            pharm_frag = pharm_frags[frag_idx]
            return torch.tensor(pharm_frag), torch.tensor(mol_frag) #x, y
    


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
        self.mol_supplier = CustomSDMolSupplier(self.mols_filepath)
        


if __name__ == "__main__":
    # Load a molecule
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    # suppl = Chem.SDMolSupplier("./raw/zinc3d_test.sdf", removeHs=False, sanitize=False, strictParsing=False)
    # mol = suppl[0]
    # # Extract pharmacophore features
    # print("\nExtracting Pharmacophore Features...")
    # dataset = SubGridsDataset(mols_filename="../../data/raw/original_benzene.sdf")
    from pharmacophore2mol.data.utils import RandomFlipMolTransform, RandomRotateMolTransform
    # dataset = SubGridsDataset(mols_filepath="../../data/raw/small_planar.sdf",
    dataset = BaseDataset(mols_filepath="../../data/raw/zinc3d_test.sdf",
                              transforms=[#RandomFlipMolTransform(),
                                          RandomRotateMolTransform()
                                          ],
                              )
    
    for i in tqdm(range(len(dataset)), desc="Testing dataset", unit="sample"):
        pharm_frag, mol_frag = dataset[i]

    # tensor_x, tensor_y = dataset[16]
    # batch = dataset[:16]
    # print(tensor_x.shape, tensor_y.shape)
