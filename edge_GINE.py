from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv


class EdgeAwareGINE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
        batch_norm: bool = True,
        cat: bool = False,   # <- empfehle False als Default
        lin: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.dropout = float(dropout)
        self.cat = bool(cat)
        self.lin = bool(lin)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        hidden_in = in_channels
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_in, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(out_channels) if batch_norm else nn.Identity())
            hidden_in = out_channels

        final_in = (in_channels + num_layers * out_channels) if self.cat else out_channels
        self.final = nn.Linear(final_in, out_channels) if self.lin else nn.Identity()

    def forward(self, x, edge_index, edge_attr, batch=None):
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))

        xs = []
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(F.relu(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            xs.append(h)

        h = torch.cat([x] + xs, dim=-1) if self.cat else xs[-1]
        return self.final(h)
