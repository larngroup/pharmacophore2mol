from collections import defaultdict
import os
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils

PHARMACOPHORE_CHANNELS = [ #COPIED FROM IMRIE: keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')
    'Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe'
]

class Pharmacophore:
    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        self.channels = {k: v for v, k in enumerate(PHARMACOPHORE_CHANNELS)}
        self.features = self._extract_pharmacophore_features()

    
    @classmethod
    def from_mol(cls, mol: Chem.Mol):
        return cls(mol)
    
    @classmethod
    def from_sdf_string(cls, sdf_string: str):
        mol = Chem.MolFromMolBlock(sdf_string)
        return cls(mol)
    
    @classmethod
    def from_mol_block(cls, mol_block: str):
        mol = Chem.MolFromMolBlock(mol_block)
        return cls(mol)
    

    

    def _extract_pharmacophore_features(self):
        #centers
        feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
        features = feature_factory.GetFeaturesForMol(self.mol)

        #directions (copied from rdkit.Chem.Features)
        for feat in features:
            if feat.GetFamily() in ['Donor', 'Acceptor', 'Aromatic']:
                pos = feat.GetPos()
                family = feat.GetFamily()
                dirs = self._get_feature_direction_vector(feat, pos, family)
                print(f"Feature: {feat.GetFamily()}, Position: {pos}, Direction Vectors: {dirs}")

        return features
    

    def _get_feature_direction_vector(self, feat, pos, family):
        # (copied from rdkit.Chem.Features.ShowFeats ShowFeats function)
        # This function is used to add directional features to the pharmacophore.
        ps = []
        dirs = []
        if family == 'Aromatic':
            ps, _ = FeatDirUtils.GetAromaticFeatVects(self.mol.GetConformer(), feat.GetAtomIds(), pos,
                                                    scale=1.0)

        elif family == 'Donor':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                FeatVectsDictMethod = {
                    1: FeatDirUtils.GetDonor1FeatVects,
                    2: FeatDirUtils.GetDonor2FeatVects,
                    3: FeatDirUtils.GetDonor3FeatVects,
                }
                featAtom = self.mol.GetAtomWithIdx(aids[0])
                numHvyNbrs = len([1 for x in featAtom.GetNeighbors() if x.GetAtomicNum() > 1])
                ps, _ = FeatVectsDictMethod[numHvyNbrs](self.mol.GetConformer(), aids, scale=1.0)

        elif family == 'Acceptor':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                FeatVectsDictMethod = {
                    1: FeatDirUtils.GetDonor1FeatVects,
                    2: FeatDirUtils.GetDonor2FeatVects,
                    3: FeatDirUtils.GetDonor3FeatVects,
                }
                featAtom = self.mol.GetAtomWithIdx(aids[0])
                numHvyNbrs = len([x for x in featAtom.GetNeighbors() if x.GetAtomicNum() > 1])
                ps, _ = FeatVectsDictMethod[numHvyNbrs](self.mol.GetConformer(), aids, scale=1.0)

        else:
            raise ValueError(f"Unsupported feature family: {family}")
        for tail, head in ps:
            vect = head - tail
            dirs.append(vect / np.linalg.norm(vect)) #i think it is already normalized in the original code, but wtv TODO: check this

        return dirs

    
    def __repr__(self):
        return f"Pharmacophore({self.mol}, {len(self.features)} features)"
    
    def get_channels(self):
        return self.channels
    
    def to_list(self):
        return [(f.GetFamily(), (f.GetPos().x, f.GetPos().y, f.GetPos().z)) for f in self.features]

    def to_dict(self, np_format=True):
        feature_dict = defaultdict(list)
        for f in self.features:
            feature_dict[f.GetFamily()].append([f.GetPos().x, f.GetPos().y, f.GetPos().z])
        if np_format:
            for key in feature_dict:
                feature_dict[key] = np.array(feature_dict[key])
        return dict(feature_dict)

    
    

if __name__ == "__main__":
    # Load a molecule
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    # suppl = Chem.SDMolSupplier("./raw/zinc3d_test.sdf", removeHs=False, sanitize=False, strictParsing=False)
    # mol = suppl[0]
    # # Extract pharmacophore features
    # print("\nExtracting Pharmacophore Features...")
    # pharmacophore = Pharmacophore.from_mol(mol)
    sdf_str = '''lig.pdb


 28 31  0  0  0  0  0  0  0  0999 V2000
  -12.3750   15.6630   41.2650 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2610   14.3660   39.8260 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.1090   14.4140   41.0570 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.4740   15.2840   42.0050 N   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1510   15.2100   40.0160 N   0  0  0  0  0  0  0  0  0  0  0  0
  -16.6176   14.1407   40.7369 C   0  0  0  0  0  0  0  0  0  0  0  0
  -15.3520   13.6930   41.3060 C   0  0  0  0  0  0  0  0  0  0  0  0
  -16.4718   11.9054   42.3009 C   0  0  0  0  0  0  0  0  0  0  0  0
  -17.7694   13.3442   41.0488 C   0  0  0  0  0  0  0  0  0  0  0  0
  -15.3321   12.5656   42.0833 N   0  0  0  0  0  0  0  0  0  0  0  0
  -17.6782   12.2407   41.8231 N   0  0  0  0  0  0  0  0  0  0  0  0
  -16.3982   10.8282   43.0485 N   0  0  0  0  0  0  0  0  0  0  0  0
  -17.2100   10.3051   43.2395 H   0  0  0  0  0  0  0  0  0  0  0  0
  -15.5311   10.5438   43.4180 H   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2773   14.5917   44.3887 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5557   14.9516   45.8687 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.4526   17.4692   45.3119 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2634   17.1035   43.8042 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.8090   15.6920   43.4140 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.0728   16.3354   46.2392 N   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0543   16.3036   46.2844 H   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2602   13.6411   37.6205 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2997   12.7333   36.5337 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.4040   11.6421   36.4989 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4709   11.4518   37.5318 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4272   12.3524   38.6174 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.3220   13.4750   38.6950 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.4377   10.7705   35.4697 F   0  0  0  0  0  0  0  0  0  0  0  0
 15 16  1  0  0  0
 15 19  1  0  0  0
 16 20  1  0  0  0
 17 18  1  0  0  0
 17 20  1  0  0  0
 18 19  1  0  0  0
 19  4  1  0  0  0
 22 23  2  0  0  0
 22 27  1  0  0  0
 23 24  1  0  0  0
 24 25  2  0  0  0
 24 28  1  0  0  0
 25 26  1  0  0  0
 26 27  2  0  0  0
 27  2  1  0  0  0
  1  5  2  0  0  0
  1  4  1  0  0  0
  2  3  2  0  0  0
  2  5  1  0  0  0
  3  4  1  0  0  0
  3  7  1  0  0  0
  6  7  2  0  0  0
  6  9  1  0  0  0
  7 10  1  0  0  0
  8 11  1  0  0  0
  8 12  1  0  0  0
  8 10  2  0  0  0
  9 11  2  0  0  0
 20 21  1  0  0  0
 12 13  1  0  0  0
 12 14  1  0  0  0
M  END
> <minimizedAffinity>
0.00000

> <minimizedRMSD>
0.64667

$$$$'''
    
    pharmacophore = Pharmacophore.from_sdf_string(sdf_str)
    print(pharmacophore.to_dict())
    # print(pharmacophore.to_dict())
    # print(pharmacophore.features)