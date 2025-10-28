"""
Metrics evaluator module for generated molecules.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict
import json
import logging
from tabulate import tabulate
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from pharmacophore2mol.data.utils import get_mol_supplier
from functools import reduce, partial, lru_cache

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """
    Results from quality metrics evaluation.
    
    Attributes:
        stable_mol: Fraction of molecules with all atoms having valid valencies (0-1)
        stable_atom: Fraction of atoms with valid valencies (0-1)
        validity: Fraction passing RDKit sanitization (0-1)
        fully_connected: Fraction of fully connected molecules (0-1)
        mean_components: Average number of disconnected fragments
        max_components: Maximum number of disconnected fragments
        n_molecules: Total number of molecules evaluated
    """
    stable_mol: float
    stable_atom: float
    validity: float
    fully_connected: float
    mean_components: float
    max_components: int
    median_strain_energy: float
    n_molecules: int
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {filepath}")
    
    def print_summary(self, tablefmt: str = "grid"):
        """
        Pretty print the results using tabulate.
        
        Args:
            tablefmt: Table format for tabulate. Options include:
                     'grid', 'fancy_grid', 'simple', 'plain', 'github', 'pretty'
        """
        # Use tqdm.write to avoid breaking progress bars if they're still active
        from tqdm import tqdm
        
        summary = f"\nQuality Metrics Results (n={self.n_molecules} molecules)\n"
        
        # Prepare table data
        table_data = [
            ["Molecule Stability", f"{self.stable_mol*100:.2f}%"],
            ["Atom Stability", f"{self.stable_atom*100:.2f}%"],
            ["Validity", f"{self.validity*100:.2f}%"],
            ["Fully Connected", f"{self.fully_connected*100:.2f}%"],
            ["Mean Components", f"{self.mean_components:.2f}"],
            ["Max Components", f"{self.max_components}"],
            ["Median Strain Energy (kcal/mol)", f"{self.median_strain_energy:.2f}"],
        ]
        
        table_str = tabulate(table_data, headers=["Metric", "Value"], tablefmt=tablefmt)
        
        # Use tqdm.write for clean output even with progress bars
        tqdm.write(summary + table_str)

    def __str__(self):
        self.print_summary()
        return ""


class Evaluator:
    """
    Evaluates quality metrics for generated molecules.
    This works sort of with a MapReduce paradigm, easy to extend with new metrics.
    Some metrics are cached for performance, using lru_cache. This assumes however that the
    same Mol object is not modified between calls. That's why the cache size is set to only
    1, for further safety, while still providing the intended speedup for repeated calls on
    the same molecule, during the Map phase.
    
    Computes:
    - Stability (molecule and atom level)
    - Validity (RDKit sanitization)
    - Connectivity (connected components)
    
    Example:
        >>> evaluator = Evaluator(atom_decoder=['C', 'N', 'O', 'F'])
        >>> results = evaluator.evaluate(molecules)
        >>> results.print_summary()
        >>> results.to_json('results.json')
    """
    
    def __init__(self):
        """
        Initialize the evaluator.
        """
        #self.metrics is the MapReduce dictionary that holds the functions for each metric
        # the key is the metric name, the value is a tuple (map_function, reduce_function)
        self.metrics = {
            'validity': (self._is_valid, partial(self._percent_equal, target=True)),
            'stable_atom': (self._compute_stability, self._avg),
            'stable_mol': (self._compute_stability, partial(self._percent_equal, target=1.0)),
            'fully_connected': (self._count_components, partial(self._percent_equal, target=1)),
            'mean_components': (self._count_components, self._avg),
            'max_components': (self._count_components, self._max),
            'median_strain_energy': (self._compute_strain_energy, self._median)
        }

    def evaluate(self, molecules: List | str | Chem.SDMolSupplier) -> EvaluationResults:
        """
        Evaluate quality (for now) metrics for a list of molecules.
        
        Args:
            molecules: List of RDKit Mol objects, RDKit MolSupplier or path to SDF file.
        
        Returns:
            EvaluationResults object with all computed metrics
        """
        if isinstance(molecules, str):
            #is probably a file path to an sdf
            molecules = get_mol_supplier(molecules, remove_hs=False, sanitize=False, strict_parsing=False)

        logger.info(f"Evaluating {len(molecules)} molecules...")
        n_molecules = len(molecules)
        map_arrays = {metric: [None] * n_molecules for metric in self.metrics.keys()}

        for i, mol in tqdm(enumerate(molecules), total=n_molecules):
            for metric, (map_func, _) in self.metrics.items():
                result = map_func(mol)
                map_arrays[metric][i] = result

        # Now reduce
        reduced_results = {}
        for metric, (_, reduce_func) in self.metrics.items():
            reduced_results[metric] = reduce_func(map_arrays[metric])

        return EvaluationResults(
            stable_mol=reduced_results.get('stable_mol', 0.0),
            stable_atom=reduced_results.get('stable_atom', 0.0),
            validity=reduced_results.get('validity', 0.0),
            fully_connected=reduced_results.get('fully_connected', 0.0),
            mean_components=reduced_results.get('mean_components', 0.0),
            max_components=reduced_results.get('max_components', 0),
            median_strain_energy=reduced_results.get('median_strain_energy', 0.0),
            n_molecules=n_molecules
        )

    ######## Map functions #######

    # @lru_cache(maxsize=1)
    def _is_valid(self, mol: Chem.Mol) -> bool:
        """
        Check if mol is valid.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        return mol is not None and mol.GetNumAtoms() > 0

    @lru_cache(maxsize=1)
    def _count_components(self, mol: Chem.Mol) -> int:
        """
        Count number of connected components in the molecule.
        
        Returns:
            int: Number of connected components.
        """
        if mol is None:
            return 0
        frags = Chem.GetMolFrags(mol, asMols=False)
        return len(frags)
    
    @lru_cache(maxsize=1)
    def _compute_stability(self, mol: Chem.Mol) -> float:
        """
        Compute fraction of stable atoms in the molecule.
        By stable we mean atoms with valid valencies.
        
        Returns:
            float: Fraction of stable atoms (0-1).
        """
        if mol is None or mol.GetNumAtoms() == 0:
            return 0.0
        
        # dictionary of allowed valencies per atom type and charge
        # from MiDi/VoxMol implementation
        allowed_bonds = {
            'H': {0: 1, 1: 0, -1: 0},
            'C': {0: [3, 4], 1: 3, -1: 3},
            'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},
            'O': {0: 2, 1: 3, -1: 1},
            'F': {0: 1, -1: 0},
            'B': 3,
            'Al': 3,
            'Si': 4,
            'P': {0: [3, 5], 1: 4},
            'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
            'Cl': 1,
            'As': 3,
            'Br': {0: 1, 1: 2},
            'I': 1,
            'Hg': [1, 2],
            'Bi': [3, 5],
            'Se': [2, 4, 6]
        }
        
        n_stable_atoms = 0
        n_total_atoms = mol.GetNumAtoms()
        
        for atom in mol.GetAtoms():
            # Get atom properties
            atom_symbol = atom.GetSymbol()
            charge = atom.GetFormalCharge()
            
            # Calculate valency (sum of bond orders)
            valency = 0
            for bond in atom.GetBonds():
                bond_type = bond.GetBondType()
                if bond_type == Chem.BondType.SINGLE:
                    valency += 1
                elif bond_type == Chem.BondType.DOUBLE:
                    valency += 2
                elif bond_type == Chem.BondType.TRIPLE:
                    valency += 3
                elif bond_type == Chem.BondType.AROMATIC:
                    valency += 1.5
                # ignoring other bond types (like DATIVE)
            
            if atom_symbol not in allowed_bonds:
                # Unknown atom type, assume unstable
                continue
            
            possible_bonds = allowed_bonds[atom_symbol]
            
            is_stable = False
            
            if isinstance(possible_bonds, int):
                is_stable = (valency == possible_bonds)
                
            elif isinstance(possible_bonds, list):
                is_stable = (valency in possible_bonds)
                
            elif isinstance(possible_bonds, dict):
                # charge-dependent valencies (e.g., H, C, N, O)
                if charge in possible_bonds:
                    expected_bonds = possible_bonds[charge]
                else:
                    # use neutral as default
                    expected_bonds = possible_bonds.get(0, None)
                
                if expected_bonds is None:
                    is_stable = False
                elif isinstance(expected_bonds, int):
                    is_stable = (valency == expected_bonds)
                elif isinstance(expected_bonds, list):
                    is_stable = (valency in expected_bonds)
            
            if is_stable:
                n_stable_atoms += 1
            # else:
            #     print(f"Unstable atom: {atom_symbol} (charge: {charge}, valency: {valency})")
        
        return n_stable_atoms / n_total_atoms if n_total_atoms > 0 else 0.0
    
    # @lru_cache(maxsize=1)
    def _compute_strain_energy(self, mol: Chem.Mol) -> float:
        """
        Calculate strain energy using UFF (from PoseCheck implementation).
        
        Strain energy = Energy(generated pose) - Energy(relaxed pose)

        Technically, energy values could be different due to complexation,
        with a target for example, and not just the molecule alone. Maybe
        some conformation in the free form that isn't fully relaxed becomes
        the most stable when in complex. However, the point of this model is
        to abstract entirely from the idea of complexation, and ideally generate
        the most relaxed molecule possible from the pharmacophore alone.
        This is to say that, while energy values are not an absolute measure of strain,
        the difference between generated and relaxed should still be a good indicator,
        especially across runs.
        
        Returns:
            Strain energy in kcal/mol, or None if computation fails
        """
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        
        try:
            # Make a copy to avoid modifying original
            mol_copy = Chem.Mol(mol)
            
            # Get energy before optimization
            # ff = AllChem.UFFGetMoleculeForceField(mol)
            # if ff is None:
                # return None
            # energy_before = ff.CalcEnergy()
            
            # Optimize geometry (relax the structure)
            # result = AllChem.UFFOptimizeMolecule(mol_copy, maxIters=200)
            
            # Get energy after optimization
            # ff_after = AllChem.UFFGetMoleculeForceField(mol_copy)
            # if ff_after is None:
            #     return None
            # energy_after = ff_after.CalcEnergy()
            
            # Strain energy is the difference
            # return energy_before - energy_after
            
        except Exception as e:
            return None
    

    ####### Reducer functions #######
    def _percent_equal(self, values: List[bool], target) -> float:
        """
        Compute percentage of values equal to the target value in a list, safeguarding for empty lists.

        Args:
            values: List of boolean values.
            target: Target value to compare against.

        Returns:
            float: Percentage of True values (0-1).
        """
        if len(values) == 0:
            return 0.0
        for value in values:
            if value is None:
                continue
        count_equal = sum(1 for v in values if v == target)
        return count_equal / len(values)
    
    def _avg(self, values: List[float]) -> float:
        """
        Compute average of a list of float values, safeguarding for empty lists.

        Args:
            values: List of float values.

        Returns:
            float: Average value (0.0 if empty).
        """
        if len(values) == 0:
            return 0.0
        return sum(values) / len(values)
    
    def _max(self, values: List[int]) -> int:
        """
        Compute maximum of a list of integer values, safeguarding for empty lists.

        Args:
            values: List of integer values.

        Returns:
            int: Maximum value (0 if empty).
        """
        if len(values) == 0:
            return 0
        return max(values)
    
    def _median(self, values: List[float]) -> float:
        """
        Compute median of a list of float values, safeguarding for empty lists, and popping None values.

        Args:
            values: List of float values.

        Returns:
            float: Median value (0.0 if empty).
        """

        sorted_vals = sorted(v for v in values if v is not None)
        if len(sorted_vals) == 0:
            return 0.0
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        else:
            return sorted_vals[mid]

if __name__ == "__main__":
    # Example usage
    evaluator = Evaluator()
    path = "./dump/epoch150_xyz_dir/molecules_obabel.sdf"
    path2 = "./pharmacophore2mol/data/raw/zinc3d_test.sdf"
    results = evaluator.evaluate(path)

    # mol_supplier = get_mol_supplier(path, remove_hs=False, sanitize=False, strict_parsing=False)
    # results = evaluator.evaluate(mol_supplier)

    # molecules = [mol for mol in mol_supplier if mol is not None]
    # results = evaluator.evaluate(molecules)
    
    print(results)
    # print(results.to_dict())
    results.to_json('./dump/quality_metrics.json')