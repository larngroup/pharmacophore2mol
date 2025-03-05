from rdkit import Chem


class Grapher:
    """
    This class is used to generate a graph representation of a molecule.
    It does not calculate any conformers, it needs them as input.
    """

    def __init__(self):
        pass

    def graph(self, mol: Chem.Mol):
        