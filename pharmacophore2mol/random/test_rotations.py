from pharmacophore2mol.data.voxelizer import Voxelizer, get_frag_count, fragment_voxel_grid
import numpy as np
from rdkit import Chem
from pharmacophore2mol.data.pharmacophore import Pharmacophore
from pharmacophore2mol.data.utils import get_translation_vector, plot_voxel_grid_sweep, translate_mol, mol_to_atom_dict
import os




if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))

    suppl = Chem.SDMolSupplier("../data/raw/zinc3d_test.sdf", removeHs=False, sanitize=False, strictParsing=False)
    mol = suppl[0]
    translation = get_translation_vector(mol.GetConformer().GetPositions())
    mol = translate_mol(mol, translation)

    mol_v = Voxelizer(channels=["C", "H", "N", "O", "S"], resolution=0.20, mode="dry_run")
    atom_dict = mol_to_atom_dict(mol)
    dummy_grid = mol_v.voxelize(atom_dict)
    print("dummy grid shape:", dummy_grid.shape)

    #translate the molecule to the positive octant
    # mol = translate_points_to_positive(mol) #TODO: either this or use the rdkit moltransforms or simply return the translation vector that we got from voxelization (maybe this last one is more attractive as we may even include padding with it)
    pharmacophore = Pharmacophore.from_mol(mol)
    # print(pharmacophore)

    # frag_count = get_frag_count(pharmacophore)
    # print("with frag_count:", frag_count)
    print(pharmacophore.get_channels())
    pharm_v = Voxelizer(channels=pharmacophore.get_channels(), resolution=0.20, mode="gaussian")#, pooling="max", std=1.0) #defaults
    grid = pharm_v.voxelize(pharmacophore.to_dict(), force_shape=dummy_grid.shape[1:])
    # plot_voxel_grid_sweep(grid[6], title="Voxelized Pharmacophore")
    side = mol_v.distance_to_voxel(0.3)
    stride = mol_v.distance_to_voxel(0.1)
    roi_indices = mol_v.get_indexes(pharmacophore.get_coordinates())
    # print("roi_indices:", roi_indices)
    fragments = fragment_voxel_grid(grid, side, stride, roi_indices)
    print("fragment count with voxelization:", fragments.shape[0])

    frag_count = get_frag_count(dummy_grid.shape[1:], side, stride, roi_indices)
    print("fragment count with frag_count:", frag_count)
