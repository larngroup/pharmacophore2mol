
#distance units are angstroms
import torch

def get_loss(name):
    """
    Get the loss function by name.
    """
    if name == "mse":
        return torch.nn.MSELoss()
    elif name == "bce":
        return torch.nn.BCELoss()
    elif name == "bce_logits":
        return torch.nn.BCEWithLogitsLoss()
    elif name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")

config = {
    "side": 5,
    "stride": 1,
    "resolution": 0.2,
    "channels": ["C", "H", "O"],#, "N", "S"],
    "pooling": "max",
    "std": 1.0,
    "mode": "binary",
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}