from pharmacophore2mol.data.dag_dataset import Node
from pharmacophore2mol.data.nodes import SDFLoader, FilterElements, RandomFlip, RandomRotate, ExtractCoords, ExtractPharmacophore, BoxSelector, Voxelize
from dataclasses import dataclass
from pharmacophore2mol import RAW_DATA_DIR, DATA_DIR
import torch

from pharmacophore2mol.data.pharmacophore import PHARMACOPHORE_CHANNELS


@dataclass
class ConditionalInput:
    pharmacophore: torch.Tensor
    molecule: torch.Tensor



# Note For Future Self:
# ---------------------
# This DAG module framework follows the pattern of statically defining the graph at initialization time,
# unlike dynamic like nn.Module. If not, i could have simply used that. It is also similar to other template-based patterns like the ones used in Lightning.
# If you want to avoid the define-it-in-the-script pattern, or the factory function pattern, because you may want to
# expose methods related to some intermediate component of the graph (say, for example, you want to expose a .devoxelize(...)
# method for use in post processing that is heavily based on the Voxelize node for molecules, and that may even be exposed in this node)
# then you need to move to the OOP, class based pattern. In here, either you don't subclass anything, but you loose easy integration with
# the torch ecosystem and IDE support; or you subclass Dataset directly, but you loose Node specific methods unless you redefine them,
# as well as IDE integration; or you subclass Node, and you have to place all the init logic inside setup, as the Node contract doesn't
# allow __init__ overrides. Initially, this may seem like a weird pattern, but it is actually quite common in the ML ecosystem,
# even though when you instantiate it might look like you have to pass in parents.
# torch.nn.Module does not have this problem since the graph is dynamic. But I don't think that that would allow you
# easy access to stuff like length. #TODO: investigate this, and see if it worth migrating to this pattern
# For now its what we have, and it is actually quite nice. DO SUBCLASS NODE!
class ConditionalDataset(Node):
    """
    This is the conditional dataset for pharmacophore2mol.
    

    Returns
    -------
    ConditionalInput
        A dataclass containing the pharmacophore and molecule tensors.
        The pharmacophore tensor is of shape (C, D, D, D) and the molecule tensor is of shape (C', D, D, D), where C and C' are the number of channels for the pharmacophore and molecule respectively, and D is the side length of the voxel grid.
    """
    def setup(self, sdf_filepath, allowed_elements=("C", "H", "O"), side_length=24.0, resolution=0.5, mol_channels=("C", "H", "O")):
        source = SDFLoader(sdf_filepath=sdf_filepath)
        filtered = FilterElements(source, allowed_elements=allowed_elements)
        augmented = RandomRotate(RandomFlip(filtered))
        
        mol_coords = ExtractCoords(augmented, bypass_copy=True)
        pharm_coords = ExtractCoords(ExtractPharmacophore(augmented), bypass_copy=True)
        
        box_selector_mol = BoxSelector(mol_coords, side_length=side_length, mode="random")
        box_selector_pharm = box_selector_mol.clone(new_parents=pharm_coords)
        
        voxelized_mol = Voxelize([mol_coords, box_selector_mol], channels=mol_channels, resolution=resolution)
        voxelized_pharm = voxelized_mol.clone(new_parents=[pharm_coords, box_selector_pharm])
        voxelized_pharm.change_channels(PHARMACOPHORE_CHANNELS)
        
        self.configure_parents([voxelized_mol, voxelized_pharm])

    def forward(self, mol_tensor, pharm_tensor):
        return ConditionalInput(pharmacophore=pharm_tensor, molecule=mol_tensor)
    



if __name__ == "__main__":
    dataset = ConditionalDataset(
        sdf_filepath=RAW_DATA_DIR / "zinc3d_test.sdf",
        side_length=8.0,
        resolution=0.25
    )

    for i in range(len(dataset)):
        data = dataset[i]
        print(type(data))
        # print(data)
    