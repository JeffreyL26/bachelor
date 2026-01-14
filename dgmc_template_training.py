"""dgmc_template_training.py

Minimal, *out-of-the-box* DGMC training loop for the synthetic pair JSONL created by
`synthetic_pair_builder.py`.

Why this exists
---------------

The DGMC paper/library trains on *sparse correspondences* y (ground-truth matches).
Our synthetic pairs store y as per-graph **local** indices:

  y = [[src_local_idx...], [tgt_local_idx...]]

During batching, DGMC expects:
  - y[0] as global row indices within the *batched source* node tensor
  - y[1] as **local** target indices within each target graph (DGMC pads targets per batch)

The dataset+collate in `dgmc_dataset.py` performs this conversion.

This script intentionally has *no* project-specific dependencies beyond:
  - torch
  - torch_geometric
  - dgmc (deep-graph-matching-consensus)
  - your local `graph_pipeline.py` (indirectly through dgmc_dataset)

Usage (defaults are reasonable for your small synthetic set):

  python dgmc_template_training.py \
      --pairs_path data/synthetic_training_pairs50.jsonl \
      --epochs 50 --batch_size 32

You can switch DGMC into the sparse candidate variant with --k (e.g. 10).
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

from dgmc_dataset import DGMCPairJsonlDataset, collate_pairs


# ------------------------------
# DGMC import (library-first)
# ------------------------------

try:
    # deep-graph-matching-consensus package
    from dgmc.models import DGMC  # type: ignore
except Exception:
    # fallback: some environments also expose DGMC via torch_geometric
    from torch_geometric.nn import DGMC  # type: ignore


# ------------------------------
# Simple edge-aware GNN for DGMC (psi_1 / psi_2)
# ------------------------------

class EdgeAwareGINE(nn.Module):
    """Small GINE-style GNN that supports edge_attr.

    DGMC requires `psi_2` to expose `in_channels` and `out_channels` attributes.
    We provide them for both psi_1 and psi_2.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_dim = int(edge_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            # GINEConv expects an MLP "nn" that maps node features.
            mlp = nn.Sequential(
                nn.Linear(c_in, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.lin_out = nn.Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.norms:
            bn.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # edge_attr is required for GINEConv if edge_dim was set.
        if edge_attr is None:
            # Create zeros edge attributes if the upstream pipeline omitted them.
            # (Your graph_pipeline *does* create edge_attr always, including empty [0,edge_dim].)
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))

        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin_out(x)
        return x


# ------------------------------
# Training utilities
# ------------------------------


@dataclass
class SplitMetrics:
    loss: float
    acc: float
    hits1: float
    hits3: float
    num_corr: int


def _compute_hits_at_k(model: DGMC, S: torch.Tensor, y: torch.Tensor, k: int) -> float:
    try:
        h = model.hits_at_k(k, S, y)   # dgmc docs
    except TypeError:
        h = model.hits_at_k(S, y, k)   # fallback, falls andere Signatur
    return float(h.item())  if hasattr(h, "item") else float(h)



@torch.no_grad()
def _safe_num_corr(y: Optional[torch.Tensor]) -> int:
    if y is None:
        return 0
    if not isinstance(y, torch.Tensor) or y.numel() == 0:
        return 0
    return int(y.size(1))

def _to_float(x) -> float:
    return float(x.item()) if hasattr(x, "item") else float(x)


def run_epoch(
    *,
    model: DGMC,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    num_steps: int = 10,
) -> SplitMetrics:
    """Runs one epoch. If optimizer is provided -> training mode, else eval."""

    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_hits1 = 0.0
    total_hits3 = 0.0
    total_corr = 0

    for batch in loader:
        batch_s: Batch = batch["batch_s"].to(device)
        batch_t: Batch = batch["batch_t"].to(device)
        y: torch.Tensor = batch["y"].to(device)

        # DGMC sparse mode (k >= 1) should receive y during forward in *training*,
        # so it can ensure the GT correspondence is within the candidate set.
        y_for_forward = y if (is_train and getattr(model, "k", -1) >= 1) else None

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        S0, SL = model(
            batch_s.x,
            batch_s.edge_index,
            getattr(batch_s, "edge_attr", None),
            batch_s.batch,
            batch_t.x,
            batch_t.edge_index,
            getattr(batch_t, "edge_attr", None),
            batch_t.batch,
            y=y_for_forward,
        )

        # Paper / library training: supervise both S0 and SL (if refinement is enabled).
        loss = model.loss(S0, y)
        S_for_metrics = S0
        if num_steps > 0:
            loss = loss + model.loss(SL, y)
            S_for_metrics = SL

        if is_train:
            loss.backward()
            optimizer.step()

        # Metrics are averaged over correspondences in y (not over batches).
        num_corr = int(y.size(1))
        acc = model.acc(S_for_metrics, y, reduction="mean")
        hits1 = _compute_hits_at_k(model, S_for_metrics, y, k=1)
        hits3 = _compute_hits_at_k(model, S_for_metrics, y, k=3)

        total_loss += float(loss.item()) * num_corr
        total_acc += _to_float(acc) * num_corr
        total_hits1 += float(hits1) * num_corr
        total_hits3 += float(hits3) * num_corr
        total_corr += num_corr

    denom = max(1, total_corr)
    return SplitMetrics(
        loss=total_loss / denom,
        acc=total_acc / denom,
        hits1=total_hits1 / denom,
        hits3=total_hits3 / denom,
        num_corr=total_corr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DGMC on synthetic JSONL pairs")
    p.add_argument(
        "--pairs_path",
        type=str,
        default=os.path.join("data", "synthetic_training_pairs.jsonl"),
        help="JSONL with synthetic pairs (output of synthetic_pair_builder.py)",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # DGMC params
    p.add_argument("--num_steps", type=int, default=10, help="DGMC refinement steps")
    p.add_argument(
        "--k",
        type=int,
        default=-1,
        help="DGMC candidate sparsification. -1 => dense (all targets). >=1 => sparse top-k.",
    )
    p.add_argument(
        "--detach",
        action="store_true",
        help="Detach scores between refinement steps (stabilizes training on some setups)",
    )

    # Psi networks
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--out_channels", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)

    # Split
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--num_workers", type=int, default=0)

    # IO
    p.add_argument(
        "--save_path",
        type=str,
        default=os.path.join("data", "dgmc_partial.pt"),
        help="Where to store the best checkpoint (by val loss)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Pairs:", args.pairs_path)

    dataset = DGMCPairJsonlDataset(
        args.pairs_path,
        undirected=True,
        use_only_positive=True,
        allow_pairs_without_y=False,
        prefer_source_smaller=True,
        strip_meta_from_data=True,
        require_one_to_one=True,
        require_full_source_coverage=False,
    )
    print(dataset.describe())

    n = len(dataset)
    if n < 2:
        raise RuntimeError("Dataset too small. Did you point to the correct JSONL?")

    n_train = max(1, int(args.train_frac * n))
    n_val = n - n_train
    if n_val == 0:
        n_train = n - 1
        n_val = 1

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
    )

    # Infer feature dimensions from one sample
    sample = dataset[0]
    in_channels = int(sample.data_s.x.size(-1))

    edge_attr = getattr(sample.data_s, "edge_attr", None)
    if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.size(1) > 0:
        edge_dim = int(edge_attr.size(1))
    else:
        # Hard fallback: your pipeline encodes edge types as one-hot with (len(EDGE_TYPES) + 1) dims.
        # In your current graph_pipeline.py: MEMA/METR/MENE/MEME + unknown => 5
        edge_dim = 5

    psi_1 = EdgeAwareGINE(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        edge_dim=edge_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    psi_2 = EdgeAwareGINE(
        in_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        edge_dim=edge_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model = DGMC(
        psi_1=psi_1,
        psi_2=psi_2,
        num_steps=args.num_steps,
        k=args.k,
        detach=args.detach,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_path = args.save_path
    os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)

    print(
        "Model dims | in_channels:", in_channels,
        "edge_dim:", edge_dim,
        "hidden:", args.hidden_channels,
        "out:", args.out_channels,
        "DGMC k:", args.k,
        "num_steps:", args.num_steps,
    )

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            num_steps=args.num_steps,
        )
        val_m = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            num_steps=args.num_steps,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_m.loss:.4f} acc {train_m.acc:.3f} h@1 {train_m.hits1:.3f} h@3 {train_m.hits3:.3f} (corr={train_m.num_corr}) | "
            f"val loss {val_m.loss:.4f} acc {val_m.acc:.3f} h@1 {val_m.hits1:.3f} h@3 {val_m.hits3:.3f} (corr={val_m.num_corr})"
        )

        if val_m.loss < best_val_loss:
            best_val_loss = val_m.loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "in_channels": in_channels,
                    "edge_dim": edge_dim,
                },
                best_path,
            )

    print("Best checkpoint saved to:", best_path)


if __name__ == "__main__":
    main()
