from rdkit import Chem
from functools import lru_cache
from time import perf_counter
import copy
import torch
import numpy as np

mol_supplier = Chem.SDMolSupplier("./dump/original.sdf")

@lru_cache(maxsize=128)
def get_mol_cached(idx):
    return mol_supplier[idx]


mol1 = get_mol_cached(0)
mol2 = get_mol_cached(0)
mol1.SetProp("test_prop", "test_value")
mol2 = get_mol_cached(0)
mol2 = Chem.Mol(mol2)
mol2.SetProp("test_prop", "modified_value")
mol1 = get_mol_cached(0)
print(mol2.GetProp("test_prop"))
print(mol1.GetProp("test_prop"))


def smart_copy(obj):
    """
    Creates a deep copy of specific high-perf objects efficiently.
    Falls back to deepcopy for unknown types.
    """
    # 1. RDKit Molecules (The specific C++ pointer issue)
    # The fastest way to copy a Mol is the copy constructor
    if isinstance(obj, Chem.Mol):
        return Chem.Mol(obj)
    
    # 2. PyTorch Tensors
    # We use clone() to get new memory. 
    # detatch() is implied if you want to break gradient tracking, 
    # but for data loading, clone is usually sufficient.
    elif isinstance(obj, torch.Tensor):
        return obj.clone()

    # 3. NumPy Arrays
    elif isinstance(obj, np.ndarray):
        return obj.copy()

    # 4. Lists/Tuples (Recursive check)
    # We must recurse because a list might contain Mols
    elif isinstance(obj, list):
        return [smart_copy(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(smart_copy(x) for x in obj)
    
    # 5. Dictionaries (Recursive check)
    elif isinstance(obj, dict):
        return {k: smart_copy(v) for k, v in obj.items()}

    # 6. Immutable primitives (str, int, float, bool)
    # No need to copy, safe to return as is
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    # 7. Fallback for custom objects
    else:
        raise TypeError(f"Unsupported type for smart_copy: {type(obj)}") #temporary raise
        return copy.deepcopy(obj)
    

#copy with rdkit
start_time = perf_counter()
for _ in range(10000):
    mol = smart_copy(mol_supplier[0])

end_time = perf_counter()
print(f"RDKit:\t\t{end_time - start_time} seconds")
#copy with deepcopy
start_time = perf_counter()
for _ in range(10000):
    mol = copy.deepcopy(mol_supplier[0])
end_time = perf_counter()
print(f"Deepcopy:\t{end_time - start_time} seconds")



array = np.random.rand(1000, 1000)
#copy with smart_copy
start_time = perf_counter()
for _ in range(10000):
    clone = smart_copy(array)
end_time = perf_counter()
print(f"smart:\t\t{end_time - start_time} seconds")
#copy with deepcopy
start_time = perf_counter()
for _ in range(10000):
    clone = copy.deepcopy(array)
end_time = perf_counter()
print(f"Deepcopy:\t{end_time - start_time} seconds")

tensor = torch.rand(1000, 1000)
#copy with smart_copy
start_time = perf_counter()
for _ in range(10000):
    clone = smart_copy(tensor)
end_time = perf_counter()
print(f"smart:\t\t{end_time - start_time} seconds")
#copy with deepcopy
start_time = perf_counter()
for _ in range(10000):
    clone = copy.deepcopy(tensor)
end_time = perf_counter()
print(f"Deepcopy:\t{end_time - start_time} seconds")

