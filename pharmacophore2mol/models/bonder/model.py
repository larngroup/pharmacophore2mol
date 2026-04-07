from typing import Optional

import torch
from torch import nn

from pharmacophore2mol.models.modules import MLP


class DenseBondPredictor(nn.Module):
    def __init__(
        self,
        num_atom_types: int = 120,
        atom_embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        cutoff: float = 3.5,
        min_distance: float = 0.1,
        num_classes: int = 5,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.min_distance = min_distance
        self.hidden_dim = hidden_dim

        self.atom_embedding = nn.Embedding(num_atom_types, atom_embedding_dim, padding_idx=0)
        self.atom_proj = nn.Sequential(
            nn.Linear(atom_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.edge_init = MLP(2 * hidden_dim + 1, hidden_dim)
        self.edge_layers = nn.ModuleList([
            MLP(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.node_layers = nn.ModuleList([
            MLP(2 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.edge_head = nn.Sequential(
            MLP(3 * hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def _build_masks(self, coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        pairwise_dist = torch.cdist(coords, coords, p=2)
        valid_pair = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)
        valid_pair = valid_pair & (pairwise_dist <= self.cutoff)
        valid_pair = valid_pair & (pairwise_dist > self.min_distance)
        return valid_pair, pairwise_dist

    def forward(self, coords: torch.Tensor, atomic_numbers: torch.Tensor, atom_mask: torch.Tensor):
        """
        coords: [B, N, 3]
        atomic_numbers: [B, N]
        atom_mask: [B, N]

        Returns:
            logits: [B, N, N, num_classes]
            valid_edge_mask: [B, N, N]
        """
        batch_size, num_nodes, _ = coords.shape

        atom_tokens = self.atom_embedding(atomic_numbers)
        node_states = self.atom_proj(atom_tokens)

        valid_edge_mask, pairwise_dist = self._build_masks(coords, atom_mask)
        distance_feature = (pairwise_dist / self.cutoff).clamp(0.0, 1.0).unsqueeze(-1)

        h_i = node_states.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        h_j = node_states.unsqueeze(1).expand(-1, num_nodes, -1, -1)

        edge_inputs = torch.cat([h_i, h_j, distance_feature], dim=-1)
        edge_states = self.edge_init(edge_inputs)
        edge_states = edge_states * valid_edge_mask.unsqueeze(-1)

        for edge_layer, node_layer in zip(self.edge_layers, self.node_layers):
            edge_states = edge_layer(torch.cat([h_i, h_j, edge_states], dim=-1))
            edge_states = edge_states * valid_edge_mask.unsqueeze(-1)
            messages = edge_states.sum(dim=2)
            node_states = node_layer(torch.cat([node_states, messages], dim=-1))
            node_states = node_states * atom_mask.unsqueeze(-1)
            h_i = node_states.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            h_j = node_states.unsqueeze(1).expand(-1, num_nodes, -1, -1)

        final_edge_inputs = torch.cat([h_i, h_j, edge_states], dim=-1)
        logits = self.edge_head(final_edge_inputs)
        logits = (logits + logits.transpose(1, 2)) / 2.0

        return logits, valid_edge_mask
