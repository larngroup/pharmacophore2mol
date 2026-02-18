import inspect
import functools

import numpy as np

# ======================
# Sentinel for required parameters
# ======================
_UNSET = object()

# ======================
# Decorator to merge Base + setup signatures
# ======================
def merge_init_from_setup(cls):
    base_init = cls._base_init
    user_setup = getattr(cls, "setup", None)

    # Signatures
    base_sig = inspect.signature(base_init)
    user_sig = inspect.signature(user_setup) if user_setup else inspect.Signature()

    # Remove 'self'
    base_params = list(base_sig.parameters.values())[1:]
    user_params = list(user_sig.parameters.values())[1:]

    # Merge: user params first, base after (avoid conflicts)
    merged_params = user_params + [
        p for p in base_params if p.name not in {param.name for param in user_params}
    ]

    merged_sig = inspect.Signature(
        parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + merged_params
    )

    # Real __init__ function
    def __init__(self, *args, **kwargs):
        bound = merged_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        arguments.pop("self")

        # Split arguments for base and setup
        base_kwargs = {}
        for p in base_params:
            if p.name in arguments:
                val = arguments[p.name]
                # Fail loudly if critical argument is unset
                if getattr(cls, "_required_params", set()) and p.name in cls._required_params:
                    if val is _UNSET:
                        raise RuntimeError(
                            f"Critical argument '{p.name}' was not provided "
                            f"for {cls.__name__}"
                        )
                # Replace _UNSET with default if base param is optional
                if val is _UNSET and p.default is not inspect.Parameter.empty:
                    val = p.default
                base_kwargs[p.name] = val
        base_init(self, **base_kwargs)

        if user_setup:
            user_kwargs = {p.name: arguments[p.name] for p in user_params if p.name in arguments}
            user_setup(self, **user_kwargs)

    __init__.__signature__ = merged_sig
    functools.update_wrapper(__init__, user_setup or base_init)
    setattr(cls, "__init__", __init__)
    return cls

# ======================
# Base class
# ======================
class Base:
    _required_params = {"seed"}  # example: seed must be forwarded

    def _base_init(self, *, seed=_UNSET, internal_config="default"):
        if seed is _UNSET:
            # default to randomized
            self.seed = None
            print("[Base] seed omitted → randomized default")
        else:
            self.seed = seed
            print(f"[Base] seed = {self.seed}")
        self.internal_config = internal_config
        print(f"[Base] internal_config = {self.internal_config}")

# ======================
# User subclass
# ======================
@merge_init_from_setup
class MyComponent(Base):
    # User defines arbitrary setup parameters
    def setup(self, userparam1, userparam2=10):
        self.userparam1 = userparam1
        self.userparam2 = userparam2
        print(f"[Setup] userparam1 = {userparam1}, userparam2 = {userparam2}")


# ======================
# Testing
# ======================
if __name__ == "__main__":
    # print("\n--- Full instantiation ---")
    # obj1 = MyComponent(userparam1=5, userparam2=7, seed=42, internal_config="custom")

    # print("\n--- Only required arguments ---")
    # obj2 = MyComponent(userparam1=1, seed=99)  # userparam2 uses default, internal_config default

    # print("\n--- Randomized seed default ---")
    # obj3 = MyComponent(userparam1=100, userparam2=200, seed=None)  # explicit randomized

    # print("\n--- Missing critical seed (should fail) ---")
    # try:
    #     obj4 = MyComponent(userparam1=0)  # seed omitted → RuntimeError
    # except RuntimeError as e:
    #     print(f"Caught RuntimeError: {e}")

    from torch.utils.data import DataLoader

    class DummyDataset:
        def __len__(self):
            return np.inf  # simulate a large dataset

        def __getitem__(self, idx):
            return idx
        

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch)