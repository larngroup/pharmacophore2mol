# @RaulSofia, 2025, rauljcsofia@gmail.com
# i am formatting this as a package instead of just a copy-paste module just in case
"""
Modular DAG Data Pipeline for PyTorch
=====================================
This module implements a Pull-Based, or Directed Acyclic Graph (DAG) architecture for 
building complex, non-linear PyTorch datasets. Unlike standard linear Compose pipelines,
this architecture supports branching, merging, and multiple inputs/outputs,  
while maintaining strict determinism and memory safety.

To use and create a DAG pipeline, do as follows:

Subclass ``Node`` to create custom processing nodes. These will be the operations in your pipeline.

Every operation should be deterministic and stateless, relying only on input data and parameters. If
you ignore this, it still works, you just wont be able to synchronize random operations
across branches, like applying the same rotation to a image and to its mask in parallel branches.

Every operation should implement a ``forward`` method. #TODO: expand on this


"""


import copy
import inspect
import random
from typing import Any
import warnings
import itertools

from torch.utils.data import Dataset, Sampler
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

#max possible size of a 64 bit integer
_MAX_64_BIT = (1 << 64) - 1

_UNSET = object()


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

        if cls is not base_node:
            # forward present
            if cls.forward == base_node.forward:
                raise NotImplementedError(
                    f"Node '{cls.__name__}' is missing the `forward` method.\n"
                    f"You must implement `forward(self, ...)`."
                )

            # len for source nodes
            if not instance.parents:
                if cls.__len__ == base_node.__len__:
                    raise NotImplementedError(#TODO: do they really need tho? maybe check the pass_index or the enforce_len flags
                        f"Source Node '{cls.__name__}' is missing the `__len__` method.\n"
                        f"Source nodes must implement `__len__(self)` manually."
                    )

        return instance

class Node(Dataset, metaclass=NodeMeta):
    """
    Base class for all DAG nodes.
    Implements lazy, pull-based data processing with support for branching and merging.
    It's execution flow is fetch-copy-forward, ensuring memory safety even across C++ pointers.
    Copy step can be ignored for performance if upstream nodes guarantee immutability.
    To implement a new Node, subclass this class and implement:
    1. `setup(self, ...)`: (Optional) To handle initialization parameters. DO NOT override `__init__`.
    2. `forward(self, parent1, parent2, ...)`: (Required) To define the processing logic. DO NOT override `__getitem__`. 
       It automatically receives data from parents. Additionally, if the signature contains `index` or `seed`, these are 
       automatically injected:
       - `index` (int): The global sample index being requested.
       - `seed` (int): A deterministic random seed unique to this node and sample.

    Parameters
    ----------

    parents : Node | list[Node] | dict[str, Node] | None
        Upstream node or nodes supplying data for this node. Accepts a single Node, a list of Nodes
        (ordered inputs), or a dict mapping names to Nodes (named inputs). Use None for root/source
        nodes that produce data without upstream dependencies (default).
    salt : None | int | Node, optional
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
    is_finite : bool, optional
        Allows overriding the automatic finiteness detection.
        If True, the node is treated as having a known, fixed length.
        If False, the node is treated as an infinite stream (e.g., a random generator).
        If left as _UNSET (default), it is automatically inferred from whether the `forward` method 
        needs the `index` argument and the finiteness of the parent nodes.
    **setup_kwargs : Any
        Any additional keyword arguments are passed directly to the user-defined `setup()` method.
        This allows you to define custom configuration parameters for your Node in its `setup` signature
        and pass them during initialization.
    """
    
    def __init__(self, parents=None, *, salt=None, bypass_copy=False, is_finite=_UNSET, **setup_kwargs):
        # 1. Store static configs
        self._training = True
        self._continue_on_error = False
        self.copy_inputs = not bypass_copy
        
        if isinstance(salt, Node): #salt is the unique identity of the node, not actual seed passed, although it depends on salt 
            self._salt = salt.salt
        elif isinstance(salt, int):
            self._salt = salt
        else:
            self._salt = random.randint(0, (1 << 64) - 1)

        #signature checks and magic injection
        sig = inspect.signature(self.forward)
        params = sig.parameters
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        
        self._pass_seed = has_kwargs or 'seed' in params
        self._pass_index = has_kwargs or 'index' in params
        self._forward_sig = sig # cache signature for validation

        self._is_finite_explicit = is_finite
        self.configure_parents(parents)

        self.setup(**setup_kwargs) #user init hook
        self._is_initialized = True

    @property
    def is_finite(self):
        """
        Dynamically infers whether the node represents a finite dataset.
        """
        if getattr(self, '_is_finite_explicit', _UNSET) is not _UNSET:
            return self._is_finite_explicit
            
        if self._pass_index:
            return True
            
        if not getattr(self, 'parents', None):
            base_len = Node.__len__
            user_len = self.__class__.__len__
            return user_len is not base_len
            
        parents_iterable = []
        if isinstance(self.parents, list):
            parents_iterable = self.parents
        elif isinstance(self.parents, dict):
            parents_iterable = self.parents.values()
        elif isinstance(self.parents, (Node, Dataset)):
            parents_iterable = [self.parents]
        
        return any(getattr(p, 'is_finite', True) for p in parents_iterable)

    def configure_parents(self, parents, is_finite_override=_UNSET):
        """
        Validates parents, calculates length/finiteness.
        Used by __init__ and clone().
        """
        if parents is not None:
            # Strict validation to enforce keyword usage for source nodes
            is_valid_parent = False
            if isinstance(parents, (Node, Dataset)):
                is_valid_parent = True
            elif isinstance(parents, list) and all(isinstance(p, (Node, Dataset)) for p in parents):
                is_valid_parent = True
            elif isinstance(parents, dict) and all(isinstance(p, (Node, Dataset)) for p in parents.values()):
                is_valid_parent = True
            
            if not is_valid_parent:
                raise TypeError(
                    f"Invalid argument passed to 'parents' in Node '{self.__class__.__name__}'.\n"
                    f"Got type: {type(parents).__name__} ({parents!r})\n"
                    f"If you intended to pass this as a configuration parameter for 'setup()', "
                    f"you MUST pass it as a keyword argument (e.g., param={parents!r}).\n"
                    f"Positional arguments are strictly reserved for parent nodes.\n"
                    f"Correct Usage:\n"
                    f"  Source Node:  {self.__class__.__name__}(filepath='data.csv')\n"
                    f"  Process Node: {self.__class__.__name__}(parent_node, factor=10)\n"
                )

        self.parents = parents if parents is not None else []

        if is_finite_override is not _UNSET:
            self._is_finite_explicit = is_finite_override

        self._validate_forward_signature(self._forward_sig)

    def __deepcopy__(self, memo):
        """
        Custom deepcopy to prevent cloning the entire upstream graph (parents).
        We only want to deepcopy the Node's internal configuration/state.
        Parents are references to other nodes, and we usually want to keep pointing 
        to the SAME parents (or rebind them later), not clone a whole new pipeline of parents.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        saved_parents = self.parents
        self.parents = None 
        
        try:
            
            for k, v in self.__dict__.items():
                if k == 'parents': 
                    continue # skip parents, we set it to None manually on the new instance
                setattr(result, k, copy.deepcopy(v, memo))
                
        finally:
            self.parents = saved_parents

        result.parents = saved_parents #yeah, although we likely want to change this *after* the deepcopy, the deepcopy itself should return a perfect copy of self 
        
        return result

    def clone(self, new_parents=None, *, salt=_UNSET, bypass_copy=_UNSET, is_finite=_UNSET):
        """
        Creates a synchronized DEEP copy of this node, optionally reconfigured.
        
        This uses `copy.deepcopy`, so the new node is completely independent of the original
        regarding mutable state (lists, buffers, etc.), preventing side effects.
        
        By default, the clone shares the same random 'salt' as the original.
        This means if the node is a random augmentation (like RandomRotate),
        the clone will perform the EXACT SAME transformation as the original 
        for the same sample index. This can be changed too, see parameters below.

        Parameters
        ----------
        new_parents : Node | list[Node] | dict | None
            If provided, rebinds the node to new upstream data sources.
            If None, retains the original parents (by reference, not cloned).
        salt : int | Node | None | _UNSET
            - _UNSET (Default): The clone shares the EXACT SAME salt as the original (Synchronized with the original node).
            - None: The clone gets a NEW RANDOM salt (Independent).
            - int/Node: The clone uses this specific salt or syncs to that specific node.
        bypass_copy : bool
            If provided, overrides the copy behavior for the clone.
        is_finite : bool
            If provided, forces the finiteness mode.
        """
        
        new_node = copy.deepcopy(self)# __deepcopy__ already handles not deepcopying the parents too (and the graph)
        
        #reconfigs only, deepcopy already handled copying
        if salt is not _UNSET: #else the original salt is already there, copied
            if salt is None:
                 new_node._salt = random.randint(0, (1 << 64) - 1)
            elif isinstance(salt, Node): 
                new_node._salt = salt.salt
            elif isinstance(salt, int):
                new_node._salt = salt
            else:
                new_node._salt = salt

        if bypass_copy is not _UNSET:
            new_node.copy_inputs = not bypass_copy
            
        # if new_parents is provided, we rebind, and this should be the main use case. Otherwise, we keep the parents set by deepcopy (which match self.parents).
        parents_to_use = new_parents if new_parents is not None else new_node.parents
        
        # if parents changed and finiteness wasn't explicit, allow re-inference
        # but if finiteness WAS explicit (is_finite arg), we pass it to configure
        finite_arg = is_finite
        if new_parents is not None and is_finite is _UNSET:
            new_node._is_finite_explicit = _UNSET
            
        new_node.configure_parents(parents_to_use, is_finite_override=finite_arg)
            
        return new_node

    def __init_subclass__(cls, **kwargs): #users should not override __init__, but setup instead
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__name__} cannot override __init__; use setup() instead")
        if "__getitem__" in cls.__dict__:
            raise TypeError(f"{cls.__name__} cannot override __getitem__; use forward() to define processing logic instead")

    def _validate_forward_signature(self, sig):
        """
        Ensures 'index' and 'seed' are not defined in positions that would 
        capture parent inputs, causing a TypeError at runtime.
        """
        num_parents = 0
        if isinstance(self.parents, list):
            num_parents = len(self.parents)
        elif isinstance(self.parents, (Node, Dataset)):
            num_parents = 1
        #if parents is a dict, inputs are passed as kwargs, so no collision risk.
        
        if num_parents == 0: #no risk
            return

        param_names = list(sig.parameters.keys())
        
        danger_zone = param_names[:num_parents] #self is already excluded automatically by inspect
        
        for i, name in enumerate(danger_zone):
            if name in ('index', 'seed'):
                raise TypeError(
                    f"\n[Signature Error] In Node '{self.__class__.__name__}', the argument '{name}' "
                    f"is defined at position {i+1} (arg '{name}'), but this slot is reserved for parent input #{i+1}.\n"
                    f"FIX: Move '{name}' to the end of the argument list.\n"
                    f"EXAMPLE: def forward(self, parent_input, ..., {name})"
                )
            
    @property
    def continue_on_error(self):
        """
        Gets the current error handling mode.
        If True, exceptions are caught and returned as FailedSample.
        """
        return self._continue_on_error
    
    @continue_on_error.setter
    def continue_on_error(self, value):
        """
        Sets the error handling mode for this node AND propagates it 
        to all upstream parent nodes recursively.
        """
        if not isinstance(value, bool):
            raise ValueError(f"'continue_on_error' must be a boolean. Got: {type(value).__name__}")
        self._set_attribute_recursive('_continue_on_error', value, set())

    @property
    def _parents_iterable(self):
        """
        Helper to get an iterable of parents regardless of the storage structure.
        """
        if isinstance(self.parents, list):
            return self.parents
        elif isinstance(self.parents, dict):
            return self.parents.values()
        elif isinstance(self.parents, (Node, Dataset)):
            return [self.parents]
        return []

    def _set_attribute_recursive(self, attr_name, value, visited):
        """
        Generic helper to set an attribute on this node and all upstream nodes recursively.
        Handles cycles in the graph by using a visited set.
        """
        if id(self) in visited:
            return
        visited.add(id(self))
        
        setattr(self, attr_name, value)
        
        for p in self._parents_iterable:
            if isinstance(p, Node):
                p._set_attribute_recursive(attr_name, value, visited)
            elif isinstance(p, Dataset):
                # try to set it on non-Node datasets if they support it
                if hasattr(p, attr_name):
                     setattr(p, attr_name, value)

    def __len__(self):
        return self._compute_length()
    
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
        context_cache['index'] = index
        if self._training:
            call_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            call_seed = 0 #on eval, we want the same seed every time for the same index
        context_cache['call_seed'] = call_seed

        try:
            ret = self._get(context_cache)
            if ret is None:
                origin = context_cache.get('none_origin', 'Unknown Node')
                return _FailedSample(index, error=f"None received from node '{origin}', likely due to filtering")
            
        except Exception as e:
            if self._continue_on_error:
                msg = (
                    f"Node '{self.__class__.__name__}' failed at index {index}. "
                    f"Suppressing error and returning FailedSample.\n"
                    f"Error details: {repr(e)}"
                )
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                return _FailedSample(index, error=e)
            raise e
        
        return ret
    
    # def __iter__(self):
    #     """
    #     Adds iterable support to the Node class.
    #     This is done to allow IterableDataset-like subclasses, like IterableNode, as well as providing streaming capability to any possible DAG (pure Node or mixed with IterableDataset-like).
        
    #     Also, no diamond graph issues cause iterators are shared and cached as tee'd iterators.
    #     """
        
    #     context = {}
    #     return self._get_stream(context)
    
    @property
    def training(self):
        """
        Read-only view of the current dataset mode.

        Defaults to True (training mode). This flag can be used by nodes to adjust their behavior accordingly.
        For example, a node might apply random augmentations only in training mode, or bypass them in eval mode.
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
    
    def setup(self, **kwargs):
        """
        User-defined setup method for custom parameters.
        This is where you should put any initialization logic that depends on user-defined parameters.
        The parameters are passed as keyword arguments from the constructor, and can be defined freely by the user.
        Example:
        ```python
        class MyNode(Node):
            def setup(self, myparam1, myparam2=10):
                self.myparam1 = myparam1
                self.myparam2 = myparam2
        ```
        """
        pass
    
    def forward(self, *args, **kwargs):
        """
        The node's processing logic (the actual operation).
        Subclasses must override this. 
        
        Parameters
        ----------
        *args :
            Positional inputs from parents (if parents is list/Node).
        **kwargs :
            Keyword inputs from parents (if parents is dict).
            
            Additionally, the following "magic" arguments are injected if they are present in the signature:
            - `seed` (int): A deterministic random seed unique to this node and sample. Strongly recommended as the sole source of randomness.
            - `index` (int): The global index of the sample being processed.

        Returns
        -------
        The processed output for this sample index, that will be delivered to downstream nodes.
        
        Example
        -------
        ```python
        class MyLoaderNode(Node):
            def forward(self, x, seed=None):
                # x is the input from the parent node. here could be a file path.
                # it could also be several inputs if there are several parents (for example, forward(..., x1, x2, ..., seed))
                file_contents = open(x).read()
                # if you need randomness, use the provided seed to create a local RNG or just use it directly
                index = seed % len(file_contents)  # example of using the seed for deterministic behavior
                return file_contents[index]
        ```

        """
        # This raises the error at runtime (as a backup), but the __init__ check should catch it first.
        raise NotImplementedError(
            f"Node '{self.__class__.__name__}' is missing the `forward` method.\n"
            f"You must implement `forward(self, ...)`."
        )
    
    def train(self):
        """
        Sets the dataset to training mode. Propagates the mode to all upstream nodes.

        This is the default mode. It just sets a boolean flag self.training = True, so that
        operations that behave differently in training vs eval can check it and adjust their behavior accordingly.
        For example, a node might apply random augmentations only in training mode, or bypass them in eval mode.
        """
        self._set_attribute_recursive('_training', True, set())

    def eval(self):
        """
        Sets the dataset to evaluation mode. Propagates the mode to all upstream nodes.
        
        Just a boolean flag self.training = False. In this mode, nodes can adjust their behavior accordingly, for example by bypassing random augmentations.
        """
        self._set_attribute_recursive('_training', False, set())

    
    def _resolve_parent(self, parent, context):
        """
        Helper to resolve fetching between Nodes (recursion) and Datasets (getitem).
        Handles caching logic as well, both for Nodes and "dumb" Datasets.
        """
        index = context['index']
        #check cache first
        cache_key = (id(parent), index)
        if cache_key in context:
            return context[cache_key]

        if isinstance(parent, Node):
            # keep the recursion and cache context alive
            result = parent._get(context)
        else:
            #standard Dataset (stop recursion, just grab data)
            result = parent[index]
            if result is None and 'none_origin' not in context:
                name_str = f" '{parent.name}'" if hasattr(parent, 'name') else ""
                context['none_origin'] = f"{parent.__class__.__name__}{name_str} (ID: {id(parent)})"

        context[cache_key] = result  #write to cache
        return result


    def _get(self, context):
        index = context['index']
        call_seed = context['call_seed']

        node_seed = self._mix_seeds(index, call_seed, self._salt) #safe mixing, breaks correlations caused by bad code in the forward method
                                   
        kwargs = {}
        if self._pass_seed:
            kwargs['seed'] = node_seed

        if self._pass_index:
            kwargs['index'] = index

        #fetch inputs from parents
        inputs = None
        f_kwargs = {} #unpack with **
        f_args = [] #unpack with *
        # list of parents (ordered)
        if isinstance(self.parents, list):
            # unpack with * so forward receives (idx, arg1, arg2...)
            for p in self.parents:
                parent_input = self._resolve_parent(p, context)
                if parent_input is None:
                    return None  # propagate None if any parent returns None
                if self.copy_inputs:
                    parent_input = smart_copy(parent_input)
                f_args.append(parent_input)
            
        # named parents (dict)
        elif isinstance(self.parents, dict):
            # unpack with ** so forward receives (idx, a=1, b=2...)
            for k, v in self.parents.items():
                 parent_input = self._resolve_parent(v, context)
                 if parent_input is None:
                    return None  # propagate None if any parent returns None
                 if self.copy_inputs:
                    parent_input = smart_copy(parent_input)
                 f_kwargs[k] = parent_input
        # single parent (linear chain)
        elif isinstance(self.parents, (Node, Dataset)):
            # direct pass-through
            parent_input = self._resolve_parent(self.parents, context)
            if parent_input is None:
                return None  # propagate None if parent returns None
            if self.copy_inputs:
                parent_input = smart_copy(parent_input)
            f_args = [parent_input]
        
        try:
            result = self.forward(*f_args, **f_kwargs, **kwargs)
        except Exception as e:
            name_str = f" '{self.name}'" if hasattr(self, 'name') else ""
            context_info_msg = f"Error in Node '{self.__class__.__name__}'{name_str} (Salt/ID: {self._salt}) at index {index}.\n"
            if len(e.args) > 0:
                e.args = (context_info_msg + "\n" + str(e.args[0]),) + e.args[1:]
            else:
                e.args = (context_info_msg,)
            raise e

        if result is None and 'none_origin' not in context:
            name_str = f" '{self.name}'" if hasattr(self, 'name') else ""
            context['none_origin'] = f"{self.__class__.__name__}{name_str} (Salt/ID: {self._salt})"

        return result
    
    def _compute_length(self): #TODO: check in full
        """
        Calculates the length of this node based on its parents.
        """

        #no parents, this is a source node where len was not overriden, default len is 1 but it should be overridden by the user
        if not self.parents:
            return 1
            
        # one parent
        if isinstance(self.parents, torch.utils.data.Dataset): #and this includes Nodes too...
            return len(self.parents)
            
        # several
        elif isinstance(self.parents, (list, dict)):
            parents_list = self.parents.values() if isinstance(self.parents, dict) else self.parents
            priority_lengths = [len(p) for p in parents_list if getattr(p, 'is_finite', True)]
            if len(set(priority_lengths)) > 1:
                # raise ValueError(f"Parent length mismatch! All parents must have the same length. Got: {lengths}")
                warnings.warn(f"Parent length mismatch! All parents should ideally have the same length to avoid unexpected behavior. Got: {priority_lengths}. Assuming max length: {max(priority_lengths)}", category=UserWarning, stacklevel=2)
            # return lengths[0]
            return max(priority_lengths) if priority_lengths else 1  # if lengths differ, we take the max, as long as there's at least one parent that matters for length (meaning, it needs the index somewhere upstream of it). if there's no parent in such condition, then simply default to 1, as this node does not really depend on the index and can just repeat the same sample if lengths differ or are missing.

        raise TypeError(
            f"Invalid `parents` type in Node '{self.__class__.__name__}'.\n"
            f"Expected Node, List[Node], or Dict[str, Node].\n"
            f"Got: {type(self.parents).__name__} ({self.parents})"
        )
    

    #just as a note about seeding mechanism here, in case you think 64 bit is not enough:
    #with this design, you can effectively turn any dataset into at most 2^64 (10^19) unique samples.
    #that is already way more than any model ever trained.
    #technically, if you really wanted, a clever mechanism to support larger seeds, like 256 bit
    #could be implemented. we still need each node to receive a 64 bit seed due to library requirements
    #but we could have a master seed (>256 bit) from where we extract 64 of those bits (using the salt of the node)
    # and then mix with the salt, finally providing it to the subclass for processing.
    # This way, we would have a virtually unlimited seed space, where each node has sort of "dedicated bits" in the master seed.
    #this is really easy to implement, and the number of bits in the master seed could be parametrizable.
    #however, the gains would be basically null. with 64 bit seeds, we already have a huge seed space.
    #increasing to 256 bit generation would slow everything down as data structures would not be able to fit in registers anymore.
    #so, I intentionally kept it like this, for performance.
    @staticmethod
    def _mix_seeds(*seeds: int) -> int:
        """
        Mixes any number of 64-bit integers into one 64-bit seed.
        It uses a combination of XOR and multiplication with large primes to ensure good bit diffusion and low collision probability.
        Essentially, breaks possible slopy correlations between seeds by transforming close ones like
        101 and 102 into very different seeds (at least 50% of the bits should flip on average).
        See SplitMix64 and MurmurHash3 for details.
        Note: this is not intended to preserve information. When they mix, information is lost (several 64 bits mix into one 64 bits).
        Essentially, this just "reshuffles" the bits (deterministically), which is perfectly fine for rng seeding purposes.
        """
        k1 = 0xbf58476d1ce4e5b9 #constants from SplitMix64 and MurmurHash3
        k2 = 0x94d049bb133111eb
        

        x = 0x9e3779b97f4a7c15 #init 
        
        for s in seeds:
            x ^= s
            x = (x * k1) & 0xFFFFFFFFFFFFFFFF
            
        #this ensures the final bit distribution is uniform (50% flip probability)
        x = (x ^ (x >> 30)) * k1
        x = (x ^ (x >> 27)) * k2
        x = x ^ (x >> 31)
        
        return x & 0xFFFFFFFFFFFFFFFF


# class IterableNode(Node):
#     ...

class _FailedSample:
    """
    Wrapper for failed samples.
    This is used to return the index of the failed sample, for further recovery, like replacement or drop.
    Example situations include filtering, or, eventually, exceptions that are caught and intended to be ignored instead of causing the entire pipeline to crash.
    """

    def __init__(self, index, error=None):
        self.index = index
        self.error = error

    def __repr__(self):
        return f"_FailedSample(index={self.index}, error={repr(self.error)})"


class SizableSequentialSampler(Sampler):
    """
    A variation of SequentialSampler that allows forcing a specific number of samples.
    
    Standard `SequentialSampler` always iterates exactly `len(dataset)` times.
    This sampler iterates `max_samples` times (or infinitely if None), yielding indices 
    incrementally starting from `start_index`.

    This is particularly useful for:
    1. Infinite/Streaming datasets where `len(dataset)` is not meaningful.
    2. debugging (running a small subset of a large dataset sequentially).
    3. defining an arbitrary 'epoch' size for continuous training.
    """
    def __init__(self, data_source=None, start_index=0, max_samples=None):
        self.data_source = data_source
        self.start_index = start_index
        self.max_samples = max_samples if max_samples is not None else _MAX_64_BIT
        
    def __iter__(self):
        i = self.start_index
        count = 0
        while count < self.max_samples:
            yield i
            i += 1
            count += 1
            
    def __len__(self):
        # returns the effective length (either the explicit limit or max 64 bit)
        return self.max_samples
    

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



class ReplacerCollate:
    """
    Collate function to replace failed samples with new ones, maintaining batch size.
    This is used in the DataLoader to handle _FailedSample instances returned by nodes when continue_on_error=True or when filtering causes None propagation.
    It detects the index of the failed sample and deterministically generates a new sample index to replace it, ensuring reproducibility.
    """
    def __init__(self, dataset, max_retries=1000, same_index=False):
        self.dataset = dataset
        self.max_retries = max_retries
        self.same_index = same_index
        self.RETRY_SALT = 0xBC58476D1CE4E5B9

    def __call__(self, batch):
        repaired_batch = []
        for item in batch:
            if isinstance(item, _FailedSample):
                replacement = self._get_replacement(item.index)
                repaired_batch.append(replacement)
            else:
                repaired_batch.append(item)
        return repaired_batch

    def _get_replacement(self, failed_index):
        """
        Deterministically finds a replacement for a failed index, until max_retries is reached.
        """
        # Determine effective limit and finiteness
        limit = 1
        is_finite = getattr(self.dataset, 'is_finite', True)
        try:
            limit = len(self.dataset)
            if limit <= 1: 
                is_finite = False
        except (TypeError, NotImplementedError):
            is_finite = False
            
        INT64_MAX = (1 << 63) - 1

        for attempt in range(self.max_retries):
            if self.same_index:
                new_index = failed_index
                # WARNING: In the current implementation of Node._get, the seed depends on (index, call_seed).
                # During training, call_seed is random per call, so retrying the same index automatically 
                # gets a new seed. During eval, call_seed is 0, so retrying same index gets same result 
                # (infinite loop of failure). However, ReplacerCollate is usually for training data loaders.
            else:
                raw_seed = Node._mix_seeds(failed_index, self.RETRY_SALT, attempt)
                if is_finite:
                    new_index = raw_seed % limit
                else:
                    new_index = raw_seed & INT64_MAX
                
            try:
                sample = self.dataset[new_index]
                if not isinstance(sample, _FailedSample):
                    return sample
            except Exception:
                pass  #if the replacement also fails, try again
        
        raise RuntimeError(f"Failed to get a valid replacement for index {failed_index} after {self.max_retries} attempts.")


