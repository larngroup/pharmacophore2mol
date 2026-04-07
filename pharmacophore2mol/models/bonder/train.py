from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pharmacophore2mol.data.dag_dataset import ReplacerCollate
from pharmacophore2mol.data.hub import get_dataset
from pharmacophore2mol.experiment_utils import save_run
from pharmacophore2mol.models.bonder import (
    BonderDataset,
    DenseBondPredictor,
    bonder_collate_fn,
    weighted_bond_loss,
    macro_f1_score,
    precision_recall_per_class,
    valid_valence_rate,
)
from pharmacophore2mol.models.bonder.dataset import ReplacerBonderCollate


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor,
    tb_writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        coords = batch.coords.to(device)
        atomic_numbers = batch.atomic_numbers.to(device)
        bond_orders = batch.bond_orders.to(device)
        atom_mask = batch.atom_mask.to(device)

        optimizer.zero_grad()
        logits, edge_mask = model(coords, atomic_numbers, atom_mask)
        loss = weighted_bond_loss(logits, bond_orders, edge_mask, class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", loss.item(), epoch * len(loader) + batch_idx)

    return total_loss / len(loader)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tb_writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    all_masks = []
    
    total_valid_atoms = 0
    total_atoms_evaluated = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            coords = batch.coords.to(device)
            atomic_numbers = batch.atomic_numbers.to(device)
            bond_orders = batch.bond_orders.to(device)
            atom_mask = batch.atom_mask.to(device)

            logits, edge_mask = model(coords, atomic_numbers, atom_mask)
            loss = weighted_bond_loss(logits, bond_orders, edge_mask)
            total_loss += loss.item()
            
            rate, batch_atoms = valid_valence_rate(logits, atomic_numbers, atom_mask)
            total_valid_atoms += int(rate * batch_atoms)
            total_atoms_evaluated += batch_atoms
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Flatten to allow concatenating tensors of variable max_atoms length
            all_logits.append(logits.view(-1, logits.shape[-1]).cpu())
            all_targets.append(bond_orders.view(-1).cpu())
            all_masks.append(edge_mask.view(-1).cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    # hardcoding num_classes=5 for bonder training
    macro_f1 = macro_f1_score(logits, targets, masks, num_classes=5)
    
    valence = total_valid_atoms / total_atoms_evaluated if total_atoms_evaluated > 0 else 0.0
    pr = precision_recall_per_class(logits, targets, masks, num_classes=5)

    if tb_writer is not None:
        tb_writer.add_scalar("val/loss", total_loss / len(loader), epoch)
        tb_writer.add_scalar("val/macro_f1", macro_f1, epoch)
        tb_writer.add_scalar("val/valid_valence_rate", valence, epoch)

    return {
        "loss": total_loss / len(loader),
        "macro_f1": macro_f1,
        "valid_valence_rate": valence,
        "precision_recall": pr,
    }


def build_class_weights():
    # weights = torch.tensor([0.2, 1.0, 2.0, 4.0, 2.0], dtype=torch.float32)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    return weights


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with save_run("./runs/bonder", dry_run=False) as run:
        writer = SummaryWriter(log_dir=run.run_dir / "tensorboard")

        train_dataset = BonderDataset(
            sdf_filepath=get_dataset(config["train_sdf"]),
            allowed_elements=config["allowed_elements"],
            jitter_sigma=config["noise_sigma"],
            max_atoms=60,
        )
        val_dataset = BonderDataset(
            sdf_filepath=get_dataset(config["val_sdf"]),
            allowed_elements=config["allowed_elements"],
            jitter_sigma=0.0,
            max_atoms=60,
        )

        train_dataset.train()
        val_dataset.eval()

        train_replacer = ReplacerCollate(train_dataset)
        val_replacer = ReplacerCollate(val_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=ReplacerBonderCollate(train_dataset),
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=ReplacerBonderCollate(val_dataset),
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

        model = DenseBondPredictor(
            num_atom_types=config["max_atomic_number"] + 1, #not the number of actual atom types, but the max atomic number + 1 for embedding (sparse table, but who cares)
            atom_embedding_dim=config["atom_embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            cutoff=config["distance_cutoff"],
            min_distance=config["min_distance"],
            num_classes=5,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config["lr_step"], gamma=config["lr_gamma"]
        )
        class_weights = build_class_weights().to(device)

        for epoch in range(1, config["epochs"] + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                class_weights,
                tb_writer=writer,
                epoch=epoch,
            )
            metrics = evaluate_epoch(model, val_loader, device, tb_writer=writer, epoch=epoch)
            scheduler.step()

            print(
                f"Epoch {epoch}/{config['epochs']} ",
                f"train_loss={train_loss:.4f}",
                f"val_loss={metrics['loss']:.4f}",
                f"macro_f1={metrics['macro_f1']:.4f}",
                f"valid_valence={metrics['valid_valence_rate']:.4f}",
            )

            torch.save(model.state_dict(), run.weights_dir / f"bonder_epoch_{epoch:03d}.pt")

        writer.close()


CONFIG = {
    "train_sdf": "geom_5confs_train.sdf",
    "val_sdf": "geom_5confs_test.sdf",
    # "allowed_elements": ["C", "H", "O", "N"],
    "allowed_elements": ['Si', 'H', 'B', 'C', 'Br', 'Bi', 'Cl', 'F', 'N', 'O', 'S', 'P', 'I'],
    "noise_sigma": 0.1,
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-3,
    "lr_step": 10,
    "lr_gamma": 0.5,
    "distance_cutoff": 3.5,
    "min_distance": 0.1,
    "max_atomic_number": 100,
    "atom_embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 4,
}

if __name__ == "__main__":
    main(CONFIG)
