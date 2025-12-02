import warnings
from rdkit import Chem
from functools import lru_cache
from time import perf_counter
import copy
import torch
import numpy as np

_UNSUPPORTED_TYPE_CACHE = set()
_CUSTOM_COPIERS = {}

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
    # rdkit mol objects (the specific c++ pointer issue)
    if Chem is not None and isinstance(obj, Chem.Mol):
        return Chem.Mol(obj)
    
    # torch tensors
    # elif torch is not None and isinstance(obj, torch.Tensor):
    #     return obj.clone()

    # numpy arrays
    elif np is not None and isinstance(obj, np.ndarray):
        return obj.copy()
    
    # no need to copy immutable primitives, safe to return as is
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    obj_type = type(obj)
    if obj_type in _CUSTOM_COPIERS:
        return _CUSTOM_COPIERS[obj_type](obj)
    
    # lists/tuples/dicts  (recursive check)
    # must recurse because a list might contain copy-optimized objects like Mols
    elif isinstance(obj, list):
        return [smart_copy(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(smart_copy(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: smart_copy(v) for k, v in obj.items()}

    
    #fallback for custom objects, slow but safe
    else:
        if obj_type not in _UNSUPPORTED_TYPE_CACHE:
            _UNSUPPORTED_TYPE_CACHE.add(obj_type)
            
            msg = (
                f"\n[Performance Warning] 'smart_copy' encountered unknown type '{obj_type.__name__}' "
                f"and is falling back to slow 'copy.deepcopy'.\n"
                f"If this type is immutable you can safely set bypass_copy=True in the node constructor.\n"
                f"If not, to fix this, choose one of the following:\n"
                f"  (Recommended) Register a fast copier for safety and speed. Example:\n"
                f"  >> register_copier({obj_type.__name__}, lambda x: x.clone())\n"
                f"  (Alternative) Disable copying (bypass_copy=True) for this node and implement custom copying inside the forward method. Example:\n"
                f"  >> MyNode(..., bypass_copy=True)"
            )
            
            warnings.warn(
                msg,
                category=UserWarning,
                stacklevel=2 
            )
        return copy.deepcopy(obj)


def register_copier(cls, copier_fn):
    """
    Register a fast copy function for a custom class.
    
    Args:
        cls: The class type (e.g., MyGraphObject)
        copier_fn: A function taking one argument (the object) and returning a copy.
    
    Example:
        register_copier(MyGraph, lambda x: x.clone())
    """
    _CUSTOM_COPIERS[cls] = copier_fn

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


print("Copying large numpy array:")
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


print("Copying large tensor:")
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

register_copier(torch.Tensor, lambda x: x.clone())

print("Copying list of tensors:")
list_of_tensors = [torch.rand(100, 100) for _ in range(100)]
#copy with smart_copy
start_time = perf_counter()
for _ in range(1000):
    clone = smart_copy(list_of_tensors)
end_time = perf_counter()
print(f"smart:\t\t{end_time - start_time} seconds")
#copy with deepcopy
start_time = perf_counter()
for _ in range(1000):
    clone = copy.deepcopy(list_of_tensors)
end_time = perf_counter()
print(f"Deepcopy:\t{end_time - start_time} seconds")


print("Copying dict of numpy arrays:")
dict_of_arrays = {f"arr_{i}": np.random.rand(100, 100) for i in range(100)}
#copy with smart_copy
start_time = perf_counter()
for _ in range(1000):
    clone = smart_copy(dict_of_arrays)
end_time = perf_counter()
print(f"smart:\t\t{end_time - start_time} seconds")
#copy with deepcopy
start_time = perf_counter()
for _ in range(1000):
    clone = copy.deepcopy(dict_of_arrays)
end_time = perf_counter()
print(f"Deepcopy:\t{end_time - start_time} seconds")


