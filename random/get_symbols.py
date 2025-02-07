from rdkit import Chem
from tqdm import tqdm




if __name__ == "__main__":
    suppl = Chem.SDMolSupplier("data/zinc3d_shards.sdf", removeHs=False)
    elements = set()
    for mol in tqdm(suppl):
        if mol is not None:
            for atom in mol.GetAtoms():
                elements.add(atom.GetSymbol())

    for e in elements:
        print(e)