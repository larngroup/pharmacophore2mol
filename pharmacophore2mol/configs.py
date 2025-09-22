"""
Project configuration for pharmacophore2mol — quick guide

This module centralizes all default settings in one place. It defines
dataclasses with sensible defaults for each piece of the pipeline, separated
by concern. The top-level ExperimentConfig aggregates all sub-configs.

If you want to modify any settings, you can either:
- Modify the defaults here (not recommended, as it affects all experiments)
- Provide a YAML configuration file with only the settings you want to change
- Manually create/modify an ExperimentConfig instance in code

Example usage:

```python
# Load default config
cfg = ExperimentConfig()

# Load config from YAML file
cfg = load_experiment_config("my_config.yaml")

# Override specific settings in code
cfg.training.epochs = 200
cfg.optimizer.lr = 5e-4

# Print config summary
print_config_summary(cfg)
```
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Dataset / Voxelization
# ---------------------------------------------------------------------------

@dataclass
class VoxelConfig:
    # Units: Angstroms where applicable (matches data code comments)
    side: float = 5.0
    stride: float = 1.0
    resolution: float = 0.2
    channels: List[str] = field(default_factory=lambda: ["C", "H", "O"])  # maps to data voxelizer
    mode: str = "gaussian"  # passed to Voxelizer
    std: float = 0.5


@dataclass
class DatasetConfig:
    mols_filepath: Optional[str] = None
    padding: float = 0.0
    transforms: Optional[List[Any]] = None
    force_len: Optional[int] = None
    voxel: VoxelConfig = field(default_factory=VoxelConfig)

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

@dataclass
class DataloaderConfig:
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 4
    persistent_workers: bool = True

# ---------------------------------------------------------------------------
# Noise / Diffusion
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    schedule_type: str = "cosine"  # 'cosine' or 'linear' (used by DDPMScheduler)
    timesteps: int = 1000
    prediction_type: str = "epsilon"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class UnetConfig:
    # generic
    implementation: str = "v2"  # choices: 'v2', 'simple' (informational)
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None

    # options used by the simple UNet implementation (features list)
    features: Optional[List[int]] = field(default_factory=lambda: [32, 64, 128, 256])

    # options used by the V2 implementation
    n_internal_channels: int = 64
    ch_mults: Tuple[int, ...] = (1, 2, 2, 4)
    is_attn: Tuple[bool, ...] = (False, False, True, True)
    n_blocks: int = 2
    n_groups: int = 8

    # control flags
    use_time_embedding: bool = True

    # catch-all for future/implementation specific settings
    extra: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Optimization / Training
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    name: Optional[str] = None  # e.g. 'cosine', 'step', 'multistep'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 100
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision: bool = False
    grad_clip: Optional[float] = None
    save_every_n_epochs: int = 5
    seed: Optional[int] = None

# ---------------------------------------------------------------------------
# Logging / Checkpointing
# ---------------------------------------------------------------------------

@dataclass
class LoggingConfig:
    log_dir: str = "runs/"
    checkpoint_dir: str = "saves/"
    checkpoint_name: str = "checkpoint.pt"
    verbose: bool = True

# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    unet: UnetConfig = field(default_factory=UnetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # convenience methods
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------------------------------------------------------------------------
# YAML loading / merging helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]):
    """Recursively update nested dictionaries (in-place)."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_update(target[k], v)
        else:
            target[k] = v


def load_experiment_config(yaml_path: Optional[Union[str, Path]] = None, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file and/or overrides dict.

    Priority: overrides (highest) -> YAML file -> defaults
    """
    base = ExperimentConfig()
    config_dict = base.as_dict()

    if yaml_path is not None:
        yaml_vals = load_yaml_config(yaml_path)
        _deep_update(config_dict, yaml_vals)

    if overrides is not None:
        _deep_update(config_dict, overrides)

    # Reconstruct dataclasses from the merged dict
    # Note: this is explicit to keep types clear and avoid unexpected keys
    ds = DatasetConfig(**config_dict.get("dataset", {}))
    dl = DataloaderConfig(**config_dict.get("dataloader", {}))
    noise = NoiseConfig(**config_dict.get("noise", {}))
    # unified unet config: accept either a top-level 'unet' mapping or legacy
    # 'model_unet3d' / 'model_unet3d_v2' sections for backward compatibility.
    unet_vals = {}
    if "unet" in config_dict:
        unet_vals = config_dict.get("unet", {})
    else:
        # merge legacy keys if present
        unet_vals = {}
        unet_vals.update(config_dict.get("model_unet3d", {}))
        # values from model_unet3d_v2 override model_unet3d
        unet_vals.update(config_dict.get("model_unet3d_v2", {}))
    unet = UnetConfig(**unet_vals)
    opt = OptimizerConfig(**config_dict.get("optimizer", {}))
    sched = SchedulerConfig(**config_dict.get("scheduler", {}))
    train = TrainingConfig(**config_dict.get("training", {}))
    log = LoggingConfig(**config_dict.get("logging", {}))

    return ExperimentConfig(dataset=ds, dataloader=dl, noise=noise, unet=unet, optimizer=opt, scheduler=sched, training=train, logging=log)


def print_config_summary(cfg: ExperimentConfig):
    print("=== Experiment Configuration Summary ===")
    print(yaml.dump(cfg.as_dict(), sort_keys=False, default_flow_style=False))



if __name__ == "__main__":
    print_config_summary(ExperimentConfig())