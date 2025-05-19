
#distance units are angstroms
import torch


config = {
    "side": 5,
    "stride": 1,
    "resolution": 0.2,
    "channels": ["C", "H", "N", "O", "S"],
    "pooling": "max",
    "std": 0.5,
    "mode": "gaussian",
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}