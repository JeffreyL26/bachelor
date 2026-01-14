"""dgmc_dataset.py

PyG/DGMC dataset utilities for *synthetic training pairs*.

This module is designed to work with the JSONL output produced by
`synthetic_pair_builder.py`, i.e. one JSON object per line:

  {
    "graph_a": { ... JSON graph ... },
    "graph_b": { ... JSON graph ... },
    "label": 1,
    "y": [[src_idx0, ...], [tgt_idx0, ...]]
  }

Important: DGMC's `loss/acc/hits_at_k` operate on *batched* scores `S`.
In the dense case `S` has shape:

  [total_source_nodes_in_batch, max_target_nodes_in_batch]

Therefore, for supervision we must provide:
  - y[0]: global source row indices (0..total_source_nodes-1)
  - y[1]: *local* target indices within each pair's target graph (0..n_t-1)

The synthetic pairs store y in *per-graph local* indices for both sides.
`collate_pairs()` converts these into the batched format DGMC expects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch, Data

# Project import: converts JSON graph -> PyG Data(x, edge_index, edge_attr, ...)
import graph_pipeline as gp


TGraph = Dict[str, Any]
TPair = Dict[str, Any]


@dataclass
class PairItem:
    """One training/eval pair."""

    data_s: Data
    data_t: Data
    y_local: Optional[torch.Tensor]  # shape [2, K] with per-graph local indices
    label: int
    meta: Dict[str, Any]


def _parse_y(y: Any) -> Optional[torch.Tensor]:
    """Parse y from JSON into torch.LongTensor [2, K].

    Expected JSON form:
      y = [[src_indices...], [tgt_indices...]]
    """
    if y is None:
        return None

    if isinstance(y, torch.Tensor):
        if y.dtype != torch.long:
            y = y.long()
        if y.dim() == 2 and y.size(0) == 2:
            return y
        return None

    if not isinstance(y, (list, tuple)) or len(y) != 2:
        return None

    a, b = y
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return None
    if len(a) != len(b):
        return None
    if len(a) == 0:
        # allow empty correspondences (but training would be meaningless)
        return torch.empty((2, 0), dtype=torch.long)

    try:
        src = torch.tensor([int(x) for x in a], dtype=torch.long)
        tgt = torch.tensor([int(x) for x in b], dtype=torch.long)
    except Exception:
        return None

    return torch.stack([src, tgt], dim=0)


def _strip_to_tensors(data: Data) -> Data:
    """Keep only the tensor fields DGMC needs.

    This avoids batching large nested python objects (graph_attrs, node_ids, ...).
    """
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return out


def _validate_y_local(
    y: torch.Tensor,
    num_nodes_s: int,
    num_nodes_t: int,
    require_one_to_one: bool = True,
    require_full_source_coverage: bool = False,
) -> bool:
    """Validate that y indices are in range and optionally 1-1."""
    if y.dim() != 2 or y.size(0) != 2:
        return False
    if y.numel() == 0:
        return True

    src = y[0]
    tgt = y[1]
    if src.min().item() < 0 or src.max().item() >= num_nodes_s:
        return False
    if tgt.min().item() < 0 or tgt.max().item() >= num_nodes_t:
        return False

    if require_one_to_one:
        # (i) no duplicates
        if src.unique().numel() != src.numel():
            return False
        if tgt.unique().numel() != tgt.numel():
            return False

    if require_full_source_coverage:
        # all source nodes must be matched
        if src.numel() != num_nodes_s:
            return False

    return True


def _maybe_swap_source_target(
    data_a: Data,
    data_b: Data,
    y_local: Optional[torch.Tensor],
    prefer_source_smaller: bool,
) -> Tuple[Data, Data, Optional[torch.Tensor], bool]:
    """Optionally swap (A,B) so that |source| <= |target|.

    Returns (data_s, data_t, y_local_swapped, did_swap).
    """
    if not prefer_source_smaller:
        return data_a, data_b, y_local, False

    na = int(data_a.num_nodes)
    nb = int(data_b.num_nodes)
    if na <= nb:
        return data_a, data_b, y_local, False

    # swap
    if y_local is None:
        y_swapped = None
    else:
        y_swapped = torch.stack([y_local[1], y_local[0]], dim=0)
    return data_b, data_a, y_swapped, True


class DGMCPairJsonlDataset(torch.utils.data.Dataset):
    """Loads a JSONL of synthetic DGMC pairs."""

    def __init__(
        self,
        pairs_path: str,
        *,
        undirected: bool = True,
        use_only_positive: bool = True,
        allow_pairs_without_y: bool = False,
        prefer_source_smaller: bool = True,
        strip_meta_from_data: bool = True,
        require_one_to_one: bool = True,
        require_full_source_coverage: bool = False,
    ) -> None:
        super().__init__()

        self.pairs_path = pairs_path
        self.undirected = undirected
        self.use_only_positive = use_only_positive
        self.allow_pairs_without_y = allow_pairs_without_y
        self.prefer_source_smaller = prefer_source_smaller
        self.strip_meta_from_data = strip_meta_from_data
        self.require_one_to_one = require_one_to_one
        self.require_full_source_coverage = require_full_source_coverage

        if not os.path.exists(pairs_path):
            raise FileNotFoundError(pairs_path)

        self._pairs: List[TPair] = []
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSON on line {line_no} of {pairs_path}: {e}") from e

                label = int(p.get("label", 1))
                if self.use_only_positive and label != 1:
                    continue

                has_y = p.get("y") is not None
                if (not has_y) and (not self.allow_pairs_without_y):
                    # For DGMC training, pairs without y are not usable.
                    continue

                if "graph_a" not in p or "graph_b" not in p:
                    raise ValueError(
                        f"Pair on line {line_no} misses 'graph_a'/'graph_b' keys. "
                        "Expected synthetic_pair_builder.py output."
                    )

                self._pairs.append(p)

        if not self._pairs:
            raise ValueError(
                "No usable pairs were loaded. "
                "If your JSONL contains negatives (label=0) or y=None, "
                "set allow_pairs_without_y=True or use_only_positive=False."
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> PairItem:
        p = self._pairs[idx]

        g_a: TGraph = p["graph_a"]
        g_b: TGraph = p["graph_b"]
        label = int(p.get("label", 1))
        y_local = _parse_y(p.get("y"))

        # Convert JSON -> Data
        data_a = gp.json_graph_to_pyg(g_a, undirected=self.undirected)
        data_b = gp.json_graph_to_pyg(g_b, undirected=self.undirected)

        if self.strip_meta_from_data:
            data_a = _strip_to_tensors(data_a)
            data_b = _strip_to_tensors(data_b)

        # Optional swap to reduce compute
        data_s, data_t, y_local, did_swap = _maybe_swap_source_target(
            data_a, data_b, y_local, prefer_source_smaller=self.prefer_source_smaller
        )

        # Validate y (if present)
        if y_local is not None:
            ok = _validate_y_local(
                y_local,
                num_nodes_s=int(data_s.num_nodes),
                num_nodes_t=int(data_t.num_nodes),
                require_one_to_one=self.require_one_to_one,
                require_full_source_coverage=self.require_full_source_coverage,
            )
            if not ok:
                raise ValueError(
                    f"Invalid y on idx={idx} (swap={did_swap}): "
                    f"|V_s|={int(data_s.num_nodes)}, |V_t|={int(data_t.num_nodes)}, y.shape={tuple(y_local.shape)}"
                )

        meta: Dict[str, Any] = {
            "idx": idx,
            "did_swap": did_swap,
            "label": label,
            "graph_id_a": (g_a.get("graph_id") if isinstance(g_a, dict) else None),
            "graph_id_b": (g_b.get("graph_id") if isinstance(g_b, dict) else None),
        }
        if "aug" in p:
            meta["aug"] = p.get("aug")

        return PairItem(data_s=data_s, data_t=data_t, y_local=y_local, label=label, meta=meta)

    def describe(self, max_items: int = 3) -> str:
        """Small human-readable summary (for debugging)."""
        n = len(self)
        items = [self[i] for i in range(min(max_items, n))]
        parts = [f"DGMCPairJsonlDataset(n={n}, undirected={self.undirected}, prefer_source_smaller={self.prefer_source_smaller})"]
        for it in items:
            k = int(it.y_local.size(1)) if it.y_local is not None else 0
            parts.append(
                f"  - idx={it.meta.get('idx')} |V_s|={int(it.data_s.num_nodes)} |V_t|={int(it.data_t.num_nodes)} |y|={k} swap={it.meta.get('did_swap')}"
            )
        return "\n".join(parts)


def collate_pairs(batch: Sequence[PairItem]) -> Dict[str, Any]:
    """Collate function for torch.utils.data.DataLoader.

    Returns:
      {
        "batch_s": Batch,
        "batch_t": Batch,
        "y": LongTensor [2, K_total],  (batched format expected by DGMC)
        "label": LongTensor [B],
        "meta": List[dict]
      }
    """
    if not batch:
        raise ValueError("Empty batch")

    # Batch graphs
    batch_s = Batch.from_data_list([it.data_s for it in batch])
    batch_t = Batch.from_data_list([it.data_t for it in batch])

    # Build DGMC y: global source rows + local target cols
    y_rows: List[int] = []
    y_cols: List[int] = []
    src_offset = 0
    for it in batch:
        ns = int(it.data_s.num_nodes)
        nt = int(it.data_t.num_nodes)

        if it.y_local is None:
            raise ValueError(
                "collate_pairs encountered an item with y_local=None. "
                "For DGMC supervised training, filter such pairs (allow_pairs_without_y=False)."
            )

        y_loc = it.y_local
        if y_loc.numel() == 0:
            src_offset += ns
            continue

        # y_local[0] must be shifted by src_offset.
        src = (y_loc[0] + src_offset).tolist()
        tgt = y_loc[1].tolist()  # keep local within target graph

        # Safety: bounds
        if max(tgt, default=-1) >= nt:
            raise ValueError(f"Target index out of bounds in batch item idx={it.meta.get('idx')}: max_tgt={max(tgt)} nt={nt}")
        if max(src, default=-1) >= (src_offset + ns):
            raise ValueError(f"Source index out of bounds after offset in batch item idx={it.meta.get('idx')}")

        y_rows.extend(src)
        y_cols.extend(tgt)
        src_offset += ns

    y = torch.tensor([y_rows, y_cols], dtype=torch.long)

    labels = torch.tensor([int(it.label) for it in batch], dtype=torch.long)
    meta = [it.meta for it in batch]
    return {"batch_s": batch_s, "batch_t": batch_t, "y": y, "label": labels, "meta": meta}


if __name__ == "__main__":
    #Tester
    import random
    from torch.utils.data import DataLoader

    random.seed(42)

    base = os.path.dirname(os.path.abspath(__file__))
    pairs_path = os.path.join(base, "data", "synthetic_training_pairs.jsonl")
    ds = DGMCPairJsonlDataset(pairs_path, use_only_positive=True)
    print(ds.describe())

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_pairs)
    b = next(iter(dl))
    print("batch_s:", b["batch_s"].num_graphs, "graphs |", b["batch_s"].num_nodes, "nodes")
    print("batch_t:", b["batch_t"].num_graphs, "graphs |", b["batch_t"].num_nodes, "nodes")
    print("y:", tuple(b["y"].shape), "| labels:", b["label"].tolist())
