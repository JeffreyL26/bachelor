"""
dgmc_template_training.py

DGMC für die synthetischen Trainingspaare (so out-of-the-box wie möglich)

DGMC kriegt als Input:
  - y[0] globale Zeilenindizes innerhalb des gebatchten source-Tensors
  - y[1] lokale Zielindizes innerhalb eines Target-Graphen
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

from dgmc_dataset import DGMCPairJsonlDataset, pair_batcher


# ------------------------------
# DGMC IMPORT AUS GITHUB
# ------------------------------

try:
    # DGMC Paket
    from dgmc.models import DGMC
except Exception:
    from torch_geometric.nn import DGMC


# ------------------------------
# EDGE-AWARE GNN
# ------------------------------

class EdgeAwareGINE(nn.Module):
    """
    Ein GIN GNN mit Edges - GINE
    GIN (Graph Isomorphism Network) standardmößig nur Node-Features, GINE mit Edge FEatures
    """

    def __init__(
        self,
        in_channels: int,           #Dimension Node-Features
        hidden_channels: int,       #Dimension pro Layer
        out_channels: int,          #Dimension Ausgabe-Embeddings
        edge_dim: int,              #Dimension Edge-Features
        num_layers: int = 3,        #Anzahl GNN-Layer
        dropout: float = 0.0,       #Dropout-Rate
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        #Mindestens ein Layer
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_dim = int(edge_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        #Parameter als Attribute speichern
        self.convs = nn.ModuleList()            #GNN-Convolution-Layer
        self.norms = nn.ModuleList()            #Normalisierungslayer pro Convolution

        #num_layer viele Layer erzeugen
        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(c_in, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim)) #GINEConv LAyer hinzufügen, dass edge_attr berücksichtigt
            self.norms.append(nn.BatchNorm1d(hidden_channels))  #BatchNorm macht die Werte in einem Netzwerk-Layer gleichmäßiger
        #Lineare Projektion con hidden_channels auf out_channels
        self.lin_out = nn.Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Setzt alle Gewichte einmal zurück/initialisiert sie
        """
        for conv in self.convs:             #Initialisiert alle GINEConv-Layer neu
            conv.reset_parameters()
        for bn in self.norms:               #Initialisiert alle BatchNorm-Layer neu
            bn.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Ein Vorwärtslauf
        """
        # edge_attr ist benötigt, edge_dim ist gesetzt
        if edge_attr is None:
            #Wenn es fehlt wird eins erzeugt (Nullmatrix)
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))
        #Über Conv- und BatchNorm-Layer iterieren
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)          # Neue Node-Embeddings unter Nutzung von Kantenstruktur und Edge-Features
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin_out(x)
        return x


# ------------------------------
# TRAINING FUNKTIONEN
# ------------------------------


@dataclass
class Metriken:
    loss: float
    acc: float
    hits1: float
    hits3: float
    num_corr: int   #Anzahl Ground-Truth-Korrespondenzen


def _hits_at_k(model: DGMC, S: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """
    Berechnet die hits@k
    """
    try:
        h = model.hits_at_k(k, S, y)   #aus DGMC
    except TypeError:
        h = model.hits_at_k(S, y, k)   #Falls andere Signatur
    return float(h.item())  if hasattr(h, "item") else float(h)



@torch.no_grad()
def _number_of_correspondences(y: Optional[torch.Tensor]) -> int:
    """
    Wie viele Korrespondenzen in y enthalten sind
    """
    if y is None:
        return 0
    if not isinstance(y, torch.Tensor) or y.numel() == 0:
        return 0
    return int(y.size(1))

def _to_float(x) -> float:
    """
    Gibt Eingabe als FLoat aus
    """
    return float(x.item()) if hasattr(x, "item") else float(x)


def run_epoch(
    *,
    model: DGMC,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    num_steps: int = 10,
) -> Metriken:
    """
    Einen Epoch durchlaufen. Wenn es einen Optimizer gibt, dann Training, ansonsten Evaluierung
    """

    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_accuracy = 0.0
    total_hits1 = 0.0
    total_hits3 = 0.0
    total_corr = 0

    for batch in loader:
        batch_s: Batch = batch["batch_s"].to(device)
        batch_t: Batch = batch["batch_t"].to(device)
        y: torch.Tensor = batch["y"].to(device)

        # Im Sparse-Modus (k>=1) betrachtet DGMC pro Knoten nur die Top-k Kandidaten, im Training deshlab y mitgeben,
        # damit die korrekte Zuordnung sicher unter diesen Kandidaten ist (sonst kann das Modell sie nicht lernen)
        y_for_forward = y if (is_train and getattr(model, "k", -1) >= 1) else None

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        #Matrizen
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

        # Traininsloss berechnen (vor und nach refinement)
        loss = model.loss(S0, y)
        S_for_metrics = S0
        if num_steps > 0:
            loss = loss + model.loss(SL, y)
            S_for_metrics = SL

        if is_train:
            loss.backward()
            optimizer.step()

        # Metriken werden im Durchschnitt über Korrespondenzen in y genommen, nicht über Batches
        num_corr = int(y.size(1))
        acc = model.acc(S_for_metrics, y, reduction="mean")
        hits1 = _hits_at_k(model, S_for_metrics, y, k=1)
        hits3 = _hits_at_k(model, S_for_metrics, y, k=3)

        total_loss += float(loss.item()) * num_corr
        total_accuracy += _to_float(acc) * num_corr
        total_hits1 += float(hits1) * num_corr
        total_hits3 += float(hits3) * num_corr
        total_corr += num_corr

    denom = max(1, total_corr)
    return Metriken(
        loss=total_loss / denom,
        acc=total_accuracy / denom,
        hits1=total_hits1 / denom,
        hits3=total_hits3 / denom,
        num_corr=total_corr,
    )


def parse_args() -> argparse.Namespace:
    """
    Konsolenparamter
    """
    p = argparse.ArgumentParser(description="Train DGMC on synthetic training pairs")
    p.add_argument(
        "--pairs_path",
        type=str,
        #default=os.path.join("data", "synthetic_training_pairs.jsonl"),
        default=os.path.join("data", "synthetic_training_pairs_control.jsonl"),
        help="JSONL with synthetic pairs",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # DGMC Parameter
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

    # GNN-Encoder die Node-Embeddings machen
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--out_channels", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)

    # Split
    p.add_argument("--train_frac", type=float, default=0.9)     #Anteil an Paaren, die für Training genutzt werden
    p.add_argument("--num_workers", type=int, default=0)        #Anzahl Hintergrundprozesse, default: alles

    #Output
    p.add_argument(
        "--save_path",
        type=str,
        default=os.path.join("data", "dgmc_perm.pt"),
        help="Where to store the best checkpoint (by val loss)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reproduzierbarkeit gewährleisten
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

    dataset_groesse = len(dataset)
    if dataset_groesse < 2:
        raise RuntimeError("Dataset too small. Correct JSONL?")

    training_groesse = max(1, int(args.train_frac * dataset_groesse))
    validierung_groesse = dataset_groesse - training_groesse
    if validierung_groesse == 0:
        training_groesse = dataset_groesse - 1
        validierung_groesse = 1

    dataset_training, dataset_validierung = random_split(dataset, [training_groesse, validierung_groesse])

    train_loader = DataLoader(
        dataset_training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pair_batcher,
    )
    val_loader = DataLoader(
        dataset_validierung,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pair_batcher,
    )

    #Feature-Dimensionen aus Sample holen
    sample = dataset[0]
    in_channels = int(sample.source_data.x.size(-1))

    edge_attr = getattr(sample.source_data, "edge_attr", None)
    if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.size(1) > 0:
        edge_dim = int(edge_attr.size(1))
    else:
        #Fallback, falls Pipeline Fehler
        edge_dim = 5
    #PSI mit unserem EdgeAwareGINE setzen (Psi siehe Paper)
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

    #Adam als Optimizer (s.StackOverflow)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #Bestes Modell speichern
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
            f"val loss {val_m.loss:.4f} acc {val_m.acc:.3f} hits@1 {val_m.hits1:.3f} hits@3 {val_m.hits3:.3f} (corr={val_m.num_corr})"
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

    print("Best run saved to:", best_path)


if __name__ == "__main__":
    main()
