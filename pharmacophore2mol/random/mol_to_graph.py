import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def mol_to_graph(mol):
    atom_types = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

    #TODO: assert conformer
    conf = mol.GetConformer()
    positions = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


    data = Data(x=atom_types, pos=positions, edge_index=edge_index)

    return data

mol = Chem.MolFromSmiles("CCO")#etanol
AllChem.EmbedMolecule(mol)  #3d
graph = mol_to_graph(mol)

print(graph)
