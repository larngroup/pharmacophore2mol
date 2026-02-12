from rdkit import Chem
from time import perf_counter

mol_supplier = Chem.SDMolSupplier("./pharmacophore2mol/data/raw/zinc3d_shards.sdf")
print("Created supplier.")

start_time = perf_counter()
length = len(mol_supplier)
end_time = perf_counter()
print(f"Length: {length} molecules loaded in {end_time - start_time} seconds")

mol_supplier = Chem.SDMolSupplier("./pharmacophore2mol/data/raw/zinc3d_shards.sdf")
print("Created supplier.")

access_index = 1404150
start_time = perf_counter()
mol = mol_supplier[access_index-1]
print(mol)
end_time = perf_counter()
print(f"Accessed molecule at index {access_index} in {end_time - start_time} seconds")