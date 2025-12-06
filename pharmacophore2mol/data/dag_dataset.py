# @RaulSofia, 2024, rauljcsofia@gmail.com
# i am formatting this as a package instead of just a copy-paste module just in case
"""
Modular DAG Data Pipeline for PyTorch
=====================================
This module implements a Pull-Based, or Directed Acyclic Graph (DAG) architecture for 
building complex, non-linear PyTorch datasets. Unlike standard linear Compose pipelines,
this architecture supports branching, merging, and complex index manipulation (1-to-many splitting) 
while maintaining strict determinism and memory safety.

To use and create a DAG pipeline, do as follows:

Subclass ``Node`` to create custom processing nodes. These will be the operations in your pipeline.

Every operation should be deterministic and stateless, relying only on input data and parameters. If
you ignore this, then it will work the same, you just wont be able to synchronize random operations
across branches, like applying the same rotation to a image and to its mask in parallel branches.

Every operation should implement a ``forward`` method. #TODO: expand on this


"""


import copy
import inspect
import random
from typing import Any
import warnings

from torch.utils.data import Dataset
import torch
#optional imports for smart_copy
try:
    from rdkit import Chem
except ImportError:
    Chem = None  # RDKit is not available
try:
    import numpy as np
except ImportError:
    np = None  # NumPy is not available



#store custom copiers for specific types
_CUSTOM_COPIERS = {}

#cache of unsupported types to avoid spamming warnings
_UNSUPPORTED_TYPE_CACHE = set()


class NodeMeta(type(Dataset)):
    _root_class = None

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if mcs._root_class is None:
            mcs._root_class = cls
        return cls

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        base_node = NodeMeta._root_class
        
        # init check
        if not getattr(instance, '_is_initialized', False):
            raise RuntimeError(
                f"Node '{cls.__name__}' was not properly initialized.\n"
                f"Perhaps you overrode `__init__` but forgot to call `super().__init__(...)`."
            )

        if cls is not base_node:
            # forward present
            if cls.forward == base_node.forward:
                raise NotImplementedError(
                    f"Node '{cls.__name__}' is missing the `forward` method.\n"
                    f"You must implement `forward(self, index, ...)`."
                )

            # len for source nodes
            if not instance.parents:
                if cls.__len__ == base_node.__len__:
                    raise NotImplementedError(
                        f"Source Node '{cls.__name__}' is missing the `__len__` method.\n"
                        f"Source nodes must implement `__len__(self)` manually."
                    )

        return instance

class Node(Dataset, metaclass=NodeMeta):
    """
    Base class for all DAG nodes.
    Implements lazy, pull-based data processing with support for branching and merging.
    It's execution flow is fetch-copy-forward, ensuring memory safety across C++ pointers.
    Copy step can be ignored for performance if upstream nodes guarantee immutability.


    Parameters
    ----------

    parents : Node | list[Node] | dict[str, Node] | None
        Upstream node or nodes supplying data for this node. Accepts a single Node, a list of Nodes
        (ordered inputs), or a dict mapping names to Nodes (named inputs). Use None for root/source
        nodes that produce data without upstream dependencies.
    seed : None | int | Node, optional
        Controls randomness and reproducibility:
          - None (default): Independent per-node randomness derived from the sample index plus a
            unique node salt; results vary across runs.
          - int: Deterministic, synchronized randomness using the sample index combined with the
            fixed integer seed (useful for reproducible pipelines).
          - Node: Inherit synchronization from an upstream node so that multiple nodes share the same
            RNG state relative to sample indices.
    bypass_copy : bool, optional
        If True, skip the automatic deepcopy of data pulled from parent nodes before calling
        forward(). This improves performance but risks in-place modification of upstream data. Only set
        True when you can guarantee that upstream nodes produce immutable outputs or you accept shared
        mutable state.
    """
    
    def __init__(self, parents=None, seed=None, bypass_copy=False):
        self.parents = parents if parents is not None else []
        self._training = True  #default mode is training
        self.copy_inputs = not bypass_copy
        self._len = -1 #sentinel value

        #for non-source nodes, len should be calculated once during init, as we already have everything and can alert to errors early
        if self.parents:
            self._len = self._compute_length()

        # self._salt = random.randint(0, (1 << 64) - 1)  #unique salt for this node instance #TODO: should be copied if sync is added 
        if isinstance(seed, Node): #sync with another node (the first in the sync chain)
            self._salt = seed.salt
        elif isinstance(seed, int): #sync with others via fixed int
            self._salt = seed
        else:
            self._salt = random.randint(0, (1 << 64) - 1)  #unique salt for this node instance

        sig = inspect.signature(self.forward)
        params = sig.parameters
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        
        self._pass_seed = has_kwargs or 'seed' in params
        self._is_initialized = True

    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        """
        This is called by the DataLoader or the user to fetch a sample at `index`.
        It initializes the context cache and starts the recursive fetch process.
        """
        # This context cache could be replaced by the simpler lru_cache decorator,
        # but that would make it harder to control the cache lifetime and scope,
        # and for wtv reason context dict cache benchmarks are faster.
        
        # init
        context_cache = {}
        context_cache['sink_index'] = index
        context_cache['call_id'] = 0

        if self._training:
            context_cache['call_id'] = random.randint(0, (1 << 64) - 1)  #unique per fetch call in training mode

        return self._get(index, context_cache)
    
    def __iter__(self):
        """
        Adds iterable support to the Node class.
        This is done to allow IterableDataset-like subclasses, like IterableNode, as well as providing streaming capability to any possible DAG (pure Node or mixed with IterableDataset-like).
        """
        if self._len >= 0: #known length, map-style, random access possible
            for idx in range(len(self)):
                yield self[idx]
        else:# _len < 0 means unknown length (prolly -1), so we just keep yielding until StopIteration is raised upstream
            idx = 0
            while True:
                try:
                    yield self[idx] #TODO: maybe should not call __getitem__ here?
                    idx += 1
                except IndexError:
                    break
        
    
    @property
    def training(self):
        """
        Read-only view of the current dataset mode.
        """
        return self._training
    
    @training.setter
    def training(self, value):
        raise AttributeError(
            "You cannot set the 'training' attribute directly because it won't propagate to upstream nodes.\n"
            "Please use the `.train()` or `.eval()` methods instead."
        )
    
    @property
    def salt(self):
        """
        Read-only access to the Node's random identity.
        Useful for debugging or syncing other nodes to this one.
        """
        return self._salt
    
    @salt.setter
    def salt(self, value):
        raise AttributeError(
            "You cannot change 'salt' after initialization.\n"
            "The salt defines the node's identity. Changing it breaks synchronization with downstream nodes.\n"
            "If you wish to change the seed, please use the 'seed' parameter of __init__ instead."
        )
    
    def train(self):
        """
        Sets the dataset to training mode. Propagates the mode to all upstream nodes.

        In this mode, every fetch call to the dataset is independent, including calls to the same index.
        For example, rng-dependent ops like random augmentations will be applied independently on each fetch, even if the index is the same.
        This does not, however, break synchronization across branches if using a shared seed or Node as a seed.
        
        """
        self._training = True
        self._propagate_mode('train')

    def eval(self):
        """
        Sets the dataset to evaluation mode. Propagates the mode to all upstream nodes.

        In this mode, every fetch call to the dataset for the same index will return the same result.
        Rng-dependent ops like random augmentations will be consistent across multiple fetches of the same index.
        """
        self._training = False
        self._propagate_mode('eval')

    def _propagate_mode(self, mode: str):
        # propagate to parents
        parents = []
        if isinstance(self.parents, list):
            parents = self.parents
        elif isinstance(self.parents, dict):
            parents = self.parents.values()
        elif isinstance(self.parents, Node):
            parents = [self.parents]

        for p in parents:
            if hasattr(p, mode): #although all nodes should have it, if using non-node datasets upstream this avoids errors, like torch's Datasets
                getattr(p, mode)()

    def forward(self, index, *args, **kwargs):
        """
        The node's processing logic (the actual operation).
        Subclasses must override this. 
        
        Parameters
        ----------
        index : int
            The global sample index.
        *args :
            Positional inputs from parents (if parents is list/Node).
        **kwargs :
            Keyword inputs from parents (if parents is dict) 
            PLUS 'seed' (if using random) 
            PLUS 'offset' (if using 1-to-Many).
        """
        # This raises the error at runtime (as a backup), but the __init__ check should catch it first.
        raise NotImplementedError
    
    def _resolve_parent(self, parent, index, context):
        """
        Helper to resolve fetching between Nodes (recursion) and Datasets (getitem).
        Handles caching logic as well, both for Nodes and "dumb" Datasets.
        """
        #check cache first
        cache_key = (id(self), index)
        if cache_key in context:
            return context[cache_key]

        if isinstance(parent, Node):
            # keep the recursion and cache context alive
            result = parent._get(index, context)
        else:
            #standard Dataset (stop recursion, just grab data)
            result = parent[index]

        context[cache_key] = result  #write to cache
        return result


    def _get(self, index, context):
        
        sink_index = context.get('sink_index', index)
        call_id = context.get('call_id', 0) #sorta redundant and silent fail but wtv

        final_seed = (sink_index + self._salt + call_id) & 0xFFFFFFFFFFFFFFFF #just to truncate to 64 bits, cuz usually RNGs use 64 bit uint seeds

        kwargs = {}
        #handle seed
        if self._pass_seed:
            kwargs['seed'] = final_seed

        #fetch inputs from parents
        inputs = None
        f_kwargs = {} #unpack with **
        f_args = [] #unpack with *
        # list of parents (ordered)
        if isinstance(self.parents, list):
            # unpack with * so forward receives (idx, arg1, arg2...)
            inputs = [self._resolve_parent(p, index, context) for p in self.parents]
            if self.copy_inputs:
                inputs = [smart_copy(x) for x in inputs]
            f_args = inputs
            
        # named parents (dict)
        elif isinstance(self.parents, dict):
            # unpack with ** so forward receives (idx, a=1, b=2...)
            inputs = {k: self._resolve_parent(v, index, context) for k, v in self.parents.items()}
            if self.copy_inputs:
                inputs = {k: smart_copy(v) for k, v in inputs.items()}
            f_kwargs = inputs
            
        # single parent (linear chain)
        elif isinstance(self.parents, (Node, Dataset)):
            # direct pass-through
            inputs = self._resolve_parent(self.parents, index, context)
            if self.copy_inputs:
                inputs = smart_copy(inputs)
            f_args = [inputs]
            
        # source node has no parents
        result = self.forward(index, *f_args, **f_kwargs, **kwargs)

        return result
    
    def _compute_length(self): #TODO: check in full
        """
        Calculates the length of this node based on its parents.
        This runs ONCE during __init__.
        """
            
        # one parent
        if isinstance(self.parents, torch.utils.data.Dataset):
            return len(self.parents)
            
        # several
        elif isinstance(self.parents, (list, dict)):
            parents_list = self.parents.values() if isinstance(self.parents, dict) else self.parents
            lengths = [len(p) for p in parents_list]
            if len(set(lengths)) != 1:
                raise ValueError(f"Parent length mismatch! All parents must have the same length. Got: {lengths}")
            return lengths[0]

        raise TypeError(
            f"Invalid `parents` type in Node '{self.__class__.__name__}'.\n"
            f"Expected Node, List[Node], or Dict[str, Node].\n"
            f"Got: {type(self.parents).__name__} ({self.parents})"
        )
    


class IterableNode(Node):
    ...
    

def smart_copy(obj: Any) -> Any:
    """
    Creates a deep copy of specific high-perf objects efficiently.
    Falls back to deepcopy for unknown types, with a warning.
    """
    #priority: check if user registered a custom copier
    obj_type = type(obj)
    if obj_type in _CUSTOM_COPIERS:
        return _CUSTOM_COPIERS[obj_type](obj)

    # rdkit mol objects (the specific c++ pointer issue)
    if Chem is not None and isinstance(obj, Chem.Mol):
        return Chem.Mol(obj)
    
    # torch tensors
    elif torch is not None and isinstance(obj, torch.Tensor):
        return obj.clone()

    # numpy arrays
    elif np is not None and isinstance(obj, np.ndarray):
        return obj.copy()
    
    # no need to copy immutable primitives, safe to return as is
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    
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
    Example: register_copier(MyGraph, lambda x: x.clone())
    """
    _CUSTOM_COPIERS[cls] = copier_fn


if __name__ == "__main__":
    import time
    from torch.utils.data import Dataset as TorchDataset

    class MySourceNode(Node):
        def __init__(self, data):
            super().__init__()
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def forward(self, index):
            self.load_data()
            return self.data[index]
        
        def load_data(self):
            time.sleep(0.1)  # Simulate delay
            return self.data
        

    class MyTransformNode(Node):
        def __init__(self, parent, factor):
            super().__init__(parents=parent)
            self.factor = factor
        
        def forward(self, index, x):
            return x * self.factor
        
    class MyMergeNode(Node):
        
        def forward(self, index, x1, x2):
            return x1 + x2

    print("--- Testing Standard Node DAG ---")
    source = MySourceNode(data=[1, 2, 3, 4, 5])
    transform = MyTransformNode(source, factor=10)
    transform2 = MyTransformNode(source, factor=2)
    merge = MyMergeNode(parents=[transform, transform2])
    
    for i in range(len(transform)):
        print(f"Index {i}: {merge[i]}")

    print("\n--- Testing Torch Dataset Compatibility ---")
    
    # A "dumb" standard PyTorch Dataset (mimics ImageFolder, TensorDataset, etc.)
    class RawTorchDataset(TorchDataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            return self.data[index]

    # Initialize the raw dataset
    raw_ds = RawTorchDataset(data=[100, 200, 300])

    # Wrap it in a Node
    # The Node should detect it's not a 'Node' instance and use standard __getitem__
    # logic via _resolve_parent, while still applying caching.
    adapter_node = MyTransformNode(raw_ds, factor=0.5)

    for i in range(len(adapter_node)):
        print(f"Index {i} (Raw {raw_ds[i]} * 0.5): {adapter_node[i]}")


    print("\n--- Testing iterator support ---")

    for idx, value in enumerate(merge):
        print(f"Index {idx} via iterator: {value}")