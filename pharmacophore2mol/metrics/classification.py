from typing import Dict, Optional

import torch


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert logits.ndim >= 2 and targets.ndim == logits.ndim - 1
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_targets = targets.view(-1)
    flat_mask = mask.view(-1)

    if flat_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    filtered_logits = flat_logits[flat_mask]
    filtered_targets = flat_targets[flat_mask]
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn(filtered_logits, filtered_targets)


def macro_f1_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> float:
    predictions = torch.argmax(logits, dim=-1)
    flat_mask = mask.view(-1)
    preds = predictions.view(-1)[flat_mask]
    gold = targets.view(-1)[flat_mask]

    if preds.numel() == 0:
        return 0.0

    eps = 1e-8
    f1s = []
    for class_idx in range(num_classes):
        tp = ((preds == class_idx) & (gold == class_idx)).sum().item()
        fp = ((preds == class_idx) & (gold != class_idx)).sum().item()
        fn = ((preds != class_idx) & (gold == class_idx)).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        f1s.append(f1)

    return float(sum(f1s) / len(f1s))


def precision_recall_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    predictions = torch.argmax(logits, dim=-1)
    flat_mask = mask.view(-1)
    preds = predictions.view(-1)[flat_mask]
    gold = targets.view(-1)[flat_mask]

    eps = 1e-8
    results: Dict[int, Dict[str, float]] = {}
    for class_idx in range(num_classes):
        tp = ((preds == class_idx) & (gold == class_idx)).sum().item()
        fp = ((preds == class_idx) & (gold != class_idx)).sum().item()
        fn = ((preds != class_idx) & (gold == class_idx)).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        results[class_idx] = {
            "precision": precision,
            "recall": recall,
        }

    return results
