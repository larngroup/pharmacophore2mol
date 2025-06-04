import torch
from torch_geometric.nn import GCNConv, radius_graph
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from pharmacophore2mol.data.utils import RandomRotateMolTransform, RandomFlipMolTransform
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class SimpleGNNDenoiser(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super(SimpleGNNDenoiser, self).__init__()
        self.atom_embedding = torch.nn.Embedding(2, 3)
        self.conv1 = GCNConv(3 + 3, hidden_dim // 2)
        self.conv2 = GCNConv(hidden_dim // 2, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv5 = GCNConv(hidden_dim // 2, 3) # Output dimension is 3 for 3D coordinates


    def forward(self, data):
        coords = data.noisy_coords
        atom_types = data.atom_types
        edge_index = data.noisy_edge_index
        atom_embs = self.atom_embedding(atom_types)
        feats = torch.cat([coords, atom_embs], dim=1)  # Concatenate coordinates and atom embeddings
        x = F.relu(self.conv1(feats, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)
        return x
        

class NoisyAtomPositionsDataset(torch.utils.data.Dataset):
    def __init__(self, mols_filepath, noise_level=0.1, transforms=None, force_len=None):
        self.noise_level = noise_level
        self.mols_filepath = mols_filepath
        self.mol_supplier = Chem.SDMolSupplier(self.mols_filepath, removeHs=False, sanitize=True, strictParsing=False)
        self.transforms = transforms if transforms is not None else []
        self.force_len = force_len


    def __len__(self):
        if self.force_len is not None:
            return self.force_len
        return len(self.mol_supplier)
    

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self[i] for i in indices]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)

            # Handle out-of-bounds indices with force_len
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

            if self.force_len is not None:
                index = index % len(self.mol_supplier)

            
            
            mol = self.mol_supplier[index]
            if mol is None:
                raise ValueError(f"Invalid molecule at index {index} in {self.mols_filepath}.")
            # Apply transformations
            for transform in self.transforms:
                mol = transform(mol)

            atom_types = []
            for atom in mol.GetAtoms():
                atm_num = atom.GetAtomicNum()
                if atm_num == 6:  # Carbon
                    atom_types.append(0)  # Carbon
                elif atm_num == 1:  # Hydrogen
                    atom_types.append(1)  # Hydrogen
                else:
                    raise ValueError(f"Unsupported atom type: {atom.GetSymbol()}") #TODO: this is just a workaround for now
                
            atom_types = torch.tensor(atom_types, dtype=torch.long)  # Convert to tensor
            # Get 3D coordinates
            coords = mol.GetConformer().GetPositions()  # Get 3D coordinates
            coords = torch.tensor(coords, dtype=torch.float)  # Convert to tensor
            centroid = coords.mean(dim=0, keepdim=True)
            coords = coords - centroid  # Center the coordinates around the origin
            # Add noise to coordinates
            noise = torch.randn_like(coords) * self.noise_level #IN DIFFUSION WHE NEED THE SCHEDULER AND THE NORMALIZATION WITH THE SQUARE ROOTS OF THE NOISE LEVELS AT EACH TIMESTEP
            noisy_coords = coords + noise
            noisy_mol = Chem.Mol(mol)
            noisy_mol.GetConformer().SetPositions(noisy_coords.numpy().astype('float64'))  # Set noisy coordinates to the conformer
            target_mol = Chem.Mol(noisy_mol)
            AllChem.UFFOptimizeMolecule(target_mol)  # Optimize the molecule with UFF force field
            AllChem.AlignMol(target_mol, noisy_mol, prbCid=0, refCid=0)  # Align the noisy molecule to the original one
            target_coords = target_mol.GetConformer().GetPositions()
            target_coords = torch.tensor(target_coords, dtype=torch.float)  # Convert to tensor
            # Create edge index using radius graph
            target_edge_index = radius_graph(target_coords, r=2.0, loop=False)
            noisy_edge_index = radius_graph(noisy_coords, r=2.0, loop=False)
            # Return noisy coordinates and edge index
            data = Data(
                noisy_coords=noisy_coords,
                target_coords=target_coords,
                noisy_edge_index=noisy_edge_index,
                target_edge_index=target_edge_index,
                atom_types=atom_types,
                centroid=centroid,
                num_nodes = noisy_coords.size(0)
            )
            return data  # Return noisy coordinates, edge index, and original coordinates for comparison


def bond_length_loss(denoised_coords, original_coords, edge_index): #WARNING THIS ONLY WORKS FOR NON PERMUTABLE, NON-DYNAMIC GRAPHS.
    # edge_index: shape [2, num_edges]
    src, dst = edge_index
    denoised_dists = (denoised_coords[src] - denoised_coords[dst]).norm(dim=1)
    original_dists = (original_coords[src] - original_coords[dst]).norm(dim=1)
    return ((denoised_dists - original_dists)**2).mean()

def weighted_ble_mse(denoised_coords, original_coords, edge_index, alpha=0.5):
    ble = bond_length_loss(denoised_coords, original_coords, edge_index)
    mse = F.mse_loss(denoised_coords, original_coords)
    return alpha * ble + (1 - alpha) * mse

def force_field_aligned_loss(denoised_coords, noisy_coords, mol):
    noisy_mol = Chem.Mol(mol)
    noisy_mol.GetConformer().SetPositions(noisy_coords.numpy().astype('float64'))
    denoised_mol = Chem.Mol(mol)
    denoised_mol.GetConformer().SetPositions(denoised_coords.numpy().astype('float64'))
    target_mol = Chem.Mol(denoised_mol)
    AllChem.UFFOptimizeMolecule(target_mol)
    AllChem.AlignMol(target_mol, noisy_mol, prbCid=0, refCid=0)  # Align the denoised molecule to the noisy one
    target_coords = target_mol.GetConformer().GetPositions()
    target_coords = torch.tensor(target_coords, dtype=torch.float)
    return F.mse_loss(denoised_coords, target_coords)


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Load a molecule from an SDF file
    dataset = NoisyAtomPositionsDataset(mols_filepath="../data/raw/original_benzene.sdf", noise_level=0.2, transforms=[RandomRotateMolTransform()], force_len=1000)
    # print("Dataset length:", len(dataset))
    # print("First item in dataset:", dataset[0])

    # noisy_coords, noisy_edge_index, target_coords, target_edge_index, centroid, atom_types = dataset[0]
    # loss = F.mse_loss(noisy_coords, target_coords)
    # print("Mean Squared Absolute Error between noisy and original coordinates:", loss.item())

    model = SimpleGNNDenoiser(hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0) #TODO: this breaks the code. if you load directly from dataset it works, but i've been tinkering and now is incomplete
    for epoch in range(100):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        loop.set_description(f"Epoch {epoch+1}")
        loop.refresh()
        for batch in loop:
            # noisy_coords, noisy_edge_index, target_coords, target_edge_index, centroid, atom_types = batch
            optimizer.zero_grad()
            pred_coords = model(batch)# noisy_edge_index)
            # noise_pred = model(noisy_coords, atom_types, target_edge_index)# noisy_edge_index)
            # real_noise = noisy_coords - target_coords
            # print(real_noise)
            loss = F.mse_loss(pred_coords, batch.target_coords)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            loop.update(1)

        # noisy_coords, noisy_edge_index, coords, edge_index, centroid, atom_types = dataset[0]
        # noise = model(noisy_coords, atom_types, noisy_edge_index)
        # print(noise)
            
    sample = dataset[0]
    noisy_coords = sample.noisy_coords
    noisy_edge_index = sample.noisy_edge_index
    coords = sample.target_coords
    atom_types = sample.atom_types

    denoised_coords = model(sample)
    # print(noise)
    mol = Chem.SDMolSupplier("../data/raw/original_benzene.sdf", removeHs=False, sanitize=True, strictParsing=False)[0]
    original_benzene = Chem.Mol(mol)
    original_benzene.GetConformer().SetPositions(coords.numpy().astype('float64'))
    noisy_benzene = Chem.Mol(mol)
    noisy_benzene.GetConformer().SetPositions(noisy_coords.numpy().astype('float64'))
    denoised_benzene = Chem.Mol(mol)
    # denoised_coords = noisy_coords - noise  # Denoised coordinates
    denoised_benzene.GetConformer().SetPositions(denoised_coords.detach().numpy().astype('float64'))
    # AllChem.UFFOptimizeMolecule(denoised_benzene)
    # AllChem.AlignMol(denoised_benzene, noisy_benzene, prbCid=0, refCid=0)  # Align the denoised molecule to the noisy one

    # Save the molecules to SDF files
    Chem.MolToMolFile(original_benzene, "./saves/denoiser/original_benzene.sdf")
    Chem.MolToMolFile(noisy_benzene, "./saves/denoiser/noisy_benzene.sdf")
    Chem.MolToMolFile(denoised_benzene, "./saves/denoiser/denoised_benzene.sdf")

    

