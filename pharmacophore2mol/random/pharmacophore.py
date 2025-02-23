import collections
import os
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Draw, AllChem
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.Chem.rdMolTransforms import ComputeCentroid
import matplotlib.pyplot as plt
import nglview as nv
from pathlib import Path


# def show_mol(mol):
#     view = nv.show_rdkit(mol)
#     view._remote_call('setSize', target='Widget', args=['800px', '600px'])
#     view._remote_call('centerView', target='Widget')
#     view._remote_call('zoom', target='Widget', args=[0.9])
#     return view

def extract_pharmacophore_features(mol):
    # Create a feature factory
    feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    features = feature_factory.GetFeaturesForMol(mol)
    return features


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    suppl = Chem.SDMolSupplier("../data/test_mol.sdf", removeHs=False, sanitize=False, strictParsing=False)
    mol = suppl[0]
    # print([ f"{k}: {v}" for k, v in zip([feature.GetSymbol() for feature in mol.GetAtoms()], [coords for coords in mol.GetConformer().GetPositions()])])
    # mol = suppl[0]
    # # view = show_mol(mol)
    # # view._display_image()
    # # Extract pharmacophore features
    print("\nExtracting Pharmacophore Features...")
    features = extract_pharmacophore_features(mol)
    feature_frequency = collections.Counter(sorted([feature.GetFamily() for feature in features]))
    feature_coordinates = [feature.GetPos() for feature in features]
    print(feature_frequency)
    print(feature_coordinates)

