"""
dgmc_template_training.py (y-based, no CLI)

Ziel:
- DGMC "out-of-the-box" auf synthetischen Paaren trainieren, bei denen Ground Truth als
  y=[2,K] (Indexpaare) vorliegt und |V_s| != |V_t| erlaubt ist.

Wichtig (DGMC-Implementierung):
- DGMC.forward nutzt to_dense_batch intern. Dadurch wird die Korrespondenzmatrix S
  pro Batch-Graph gepaddet.
- Für sparse DGMC (k>=1) sollten Ground Truth Indizes `y` an forward(...) gegeben werden,
  damit GT-Kanten in die Top-k Kandidaten aufgenommen werden können.
- Für dense DGMC (k<1, z.B. -1) ist y nur für loss/metrics relevant.

Ausführen:
- Parameter im CONFIG-Block anpassen.
- Dann einfach `python dgmc_template_training.py`.

"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from dgmc_training_metrics import EpochSplitMetrics


# DGMC
try:
    from dgmc import DGMC  # type: ignore
except Exception:
    try:
        from dgmc.models import DGMC  # type: ignore
    except Exception as e:
        raise ImportError(
            "Konnte DGMC nicht importieren. Stelle sicher, dass deep-graph-matching-consensus installiert ist."
        ) from e

# Encoder (edge-aware)
from wrapper_GINE import EdgeAwareGINE

# Dataset/Collate (this file)
from dgmc_dataset import DGMCPairJsonlDataset, collate_pairs


# =============================
# CONFIG (hier editieren)
# =============================
BASE_DIR = Path(__file__).resolve().parent

PAIRS_PATH = str(BASE_DIR / "data" / "synthetic_training_pairs50.jsonl")
OUT_DIR = str(BASE_DIR / "runs" / "dgmc_y")

SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 0.0

# DGMC Hyperparameter
NUM_STEPS = 10
DGMC_K = -1          # -1 => dense (empfohlen für kleine Graphen), >=1 => sparse top-k
DETACH = False

# Dataset behavior
UNDIRECTED = True
ENFORCE_SOURCE_LE_TARGET = True
REQUIRE_1TO1 = True
REQUIRE_FULL_SOURCE_COVERAGE = False  # False = erlaubt partial supervision (K < |V_s|)

# Split
VAL_FRACTION = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# Helpers
# =============================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EpochLog:
    epoch: int
    split: str
    loss: float
    acc: float


class ResettableEdgeAwareGINE(torch.nn.Module):
    """
    DGMC ruft reset_parameters() auf psi_1/psi_2 in DGMC.reset_parameters().
    EdgeAwareGINE hat keine reset_parameters-Methode, daher Wrapper.
    """
    def __init__(self, base: EdgeAwareGINE) -> None:
        super().__init__()
        self.base = base
        # required by DGMC for psi_2:
        self.in_channels = getattr(base, "in_channels", None)
        self.out_channels = getattr(base, "out_channels", None)

    def reset_parameters(self) -> None:
        # best-effort: reset all submodules that implement reset_parameters
        for m in self.modules():
            if m is self:
                continue
            rp = getattr(m, "reset_parameters", None)
            if callable(rp):
                rp()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.base(x, edge_index, edge_attr, batch=batch)


def train_or_eval_epoch(model: DGMC, loader: DataLoader, opt: Adam | None) -> Tuple[float, float]:
    is_train = opt is not None
    model.train(is_train)

    split = EpochSplitMetrics(num_steps=model.num_steps)

    total_loss = 0.0
    total_acc = 0.0
    total_y = 0

    for batch in loader:
        x_s = batch["x_s"].to(DEVICE)
        ei_s = batch["edge_index_s"].to(DEVICE)
        ea_s = batch["edge_attr_s"]
        ea_s = ea_s.to(DEVICE) if ea_s is not None else None
        b_s = batch["batch_s"].to(DEVICE)

        x_t = batch["x_t"].to(DEVICE)
        ei_t = batch["edge_index_t"].to(DEVICE)
        ea_t = batch["edge_attr_t"]
        ea_t = ea_t.to(DEVICE) if ea_t is not None else None
        b_t = batch["batch_t"].to(DEVICE)

        y = batch["y"].to(DEVICE)

        if is_train:
            opt.zero_grad()

        # forward: pass y only if sparse training (k>=1)
        y_for_forward = y if (model.k >= 1 and model.training) else None
        S0, SL = model(x_s, ei_s, ea_s, b_s, x_t, ei_t, ea_t, b_t, y=y_for_forward)
        split.update(S0, SL, y, b_s)

        loss = model.loss(S0, y)
        if model.num_steps > 0:
            loss = loss + model.loss(SL, y)

        acc = model.acc(SL if model.num_steps > 0 else S0, y, reduction="mean")

        if is_train:
            loss.backward()
            opt.step()

        k = int(y.size(1))
        total_loss += float(loss.item()) * k
        total_acc += float(acc) * k
        total_y += k

    split_stats = split.finalize()
    mean_loss = total_loss / max(total_y, 1)
    mean_acc = total_acc / max(total_y, 1)
    return mean_loss, mean_acc, split_stats


def main() -> None:
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = DGMCPairJsonlDataset(
        pairs_path=PAIRS_PATH,
        use_only_positive=True,          # supervised training needs y
        undirected=UNDIRECTED,
        enforce_source_le_target=ENFORCE_SOURCE_LE_TARGET,
        require_1to1=REQUIRE_1TO1,
        require_full_source_coverage=REQUIRE_FULL_SOURCE_COVERAGE,
        allow_pairs_without_y=False,
    )

    # Split train/val
    n_total = len(ds)
    n_val = max(1, int(round(VAL_FRACTION * n_total)))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pairs)

    # infer dimensions from one sample
    sample = ds[0]
    in_dim = int(sample.data_s.x.size(-1))
    edge_dim = int(sample.data_s.edge_attr.size(-1)) if getattr(sample.data_s, "edge_attr", None) is not None else 0
    if edge_dim == 0:
        # still allow edge-less mode
        edge_dim = 1

    hidden = 64

    psi_1 = ResettableEdgeAwareGINE(EdgeAwareGINE(in_dim, hidden, edge_dim=edge_dim, num_layers=3, dropout=0.0, batch_norm=True, cat=False, lin=True))
    # psi_2 operates on random indicator functions of dimension R_in; DGMC requires in_channels/out_channels
    R_in = 16
    R_out = 16
    psi_2 = ResettableEdgeAwareGINE(EdgeAwareGINE(R_in, R_out, edge_dim=edge_dim, num_layers=2, dropout=0.0, batch_norm=True, cat=False, lin=True))

    model = DGMC(psi_1, psi_2, num_steps=NUM_STEPS, k=DGMC_K, detach=DETACH).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"[DGMC] device={DEVICE} | train={n_train} | val={n_val}")
    print(f"[DGMC] x_dim={in_dim}, edge_dim={edge_dim}, k={DGMC_K}, steps={NUM_STEPS}")

    # store dataset sanity
    with open(os.path.join(OUT_DIR, "pairs_sanity.json"), "w", encoding="utf-8") as f:
        json.dump(ds.describe(), f, indent=2, ensure_ascii=False)

    logs: List[EpochLog] = []
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_stats = train_or_eval_epoch(model, train_loader, opt)
        va_loss, va_acc, va_stats = train_or_eval_epoch(model, val_loader, None)
        #print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} "
            f"(full {tr_stats['full']['acc']:.4f}, partial {tr_stats['partial']['acc']:.4f}) | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} "
            f"(full {va_stats['full']['acc']:.4f}, partial {va_stats['partial']['acc']:.4f})"
        )
        logs.append(EpochLog(epoch, "train", tr_loss, tr_acc))
        logs.append(EpochLog(epoch, "val", va_loss, va_acc))

    # write csv log
    with open(os.path.join(OUT_DIR, "train_log.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "acc"])
        w.writeheader()
        for r in logs:
            w.writerow(asdict(r))

    # save model
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
    print("[DGMC] saved:", os.path.join(OUT_DIR, "model.pt"))


if __name__ == "__main__":
    main()
