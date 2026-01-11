"""
dgmc_dataset.py (standalone, y-based)

Warum existiert diese Datei?
- Du hast synthetische Trainingspaare, die Ground Truth als Indexpaare `y` liefern.
- DGMC (out-of-the-box) nutzt für Loss/Accuracy/Hits `y` als LongTensor der Form [2, K].

Wichtige DGMC-Semantik (laut DGMC-Implementierung):
- In der dichten Variante wird intern `to_dense_batch` genutzt; der Rückgabewert S_0/S_L hat
  Zeilen = alle Source-Knoten (über den Batch flach), Spalten = gepaddete Target-Knoten pro Graph.
- Daher muss in y:
  - y[0] = globale Row-Indices (über den Batch flach) → im Collate wird ein Offset addiert.
  - y[1] = *lokale* Target-Indizes innerhalb des jeweiligen Zielgraphs (kein Offset).

Erwartetes JSONL-Pair-Format (eine Zeile pro Pair):
{
  "graph_a": {...},               # Graph-JSON (nodes/edges/graph_attrs)
  "graph_b": {...},
  "label": 1|0,                   # 1 = positives Pair (soll y haben)
  "y": [[src...],[tgt...]] | null # [2,K] (Indexpaare) oder None
}

Hinweis:
- Negative Paare (label=0) sind fürs supervised DGMC-Training nur indirekt nutzbar, weil DGMC.loss
  Ground Truth benötigt. Diese Dataset-Klasse kann optional nur positive Paare laden.

"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

# torch_geometric wird für Batch.from_data_list benötigt (nur Tensor-Felder werden gebatcht).
from torch_geometric.data import Data, Batch  # type: ignore

# ---------------------------------------------------------------------
# Robust Import: graph_pipeline.json_graph_to_pyg
# ---------------------------------------------------------------------
gp = None  # type: ignore
try:
    import graph_pipeline as gp  # type: ignore
except Exception:
    try:
        this_dir = str(Path(__file__).resolve().parent)
        if this_dir not in sys.path:
            sys.path.insert(0, this_dir)
        import graph_pipeline as gp  # type: ignore
    except Exception:
        gp = None  # type: ignore

if gp is None:  # pragma: no cover
    raise ImportError(
        "Konnte graph_pipeline.py nicht importieren. "
        "Lege diese Datei ins Repo-Root (wo graph_pipeline.py liegt) oder setze PYTHONPATH entsprechend."
    )


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
@dataclass
class PairItem:
    data_s: Data
    data_t: Data
    y: Optional[torch.Tensor]          # [2,K] long
    label: int
    swapped: bool
    meta: Dict[str, Any]


# ---------------------------------------------------------------------
# Helper: JSONL Reader
# ---------------------------------------------------------------------
def _iter_jsonl(path: Path, max_lines: Optional[int] = None) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_lines is not None and i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def _num_nodes(graph_json: Dict[str, Any]) -> int:
    return int(len(graph_json.get("nodes") or []))


# ---------------------------------------------------------------------
# Helper: y parsing/validation
# ---------------------------------------------------------------------
def parse_y(raw_y: Any) -> Optional[torch.Tensor]:
    """
    Akzeptiert mehrere Encodings und gibt y als LongTensor [2,K] zurück.

    Unterstützt:
    - None
    - [[src...],[tgt...]]
    - {"row":[...], "col":[...]} oder {"src":[...],"dst":[...]}
    - Liste von Paaren [[s0,t0],[s1,t1],...]
    """
    if raw_y is None:
        return None

    if isinstance(raw_y, dict):
        row = raw_y.get("row", raw_y.get("src"))
        col = raw_y.get("col", raw_y.get("dst"))
        if row is None or col is None:
            raise ValueError(f"Unsupported y dict keys: {list(raw_y.keys())}")
        return torch.tensor([row, col], dtype=torch.long)

    if isinstance(raw_y, list):
        if len(raw_y) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        # [[src...],[tgt...]]
        if len(raw_y) == 2 and isinstance(raw_y[0], list) and isinstance(raw_y[1], list):
            return torch.tensor(raw_y, dtype=torch.long)

        # [[s,t], ...]
        if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw_y):
            src = [int(x[0]) for x in raw_y]
            tgt = [int(x[1]) for x in raw_y]
            return torch.tensor([src, tgt], dtype=torch.long)

    raise ValueError(f"Unsupported y format: {type(raw_y)}")


def validate_y(y: torch.Tensor, ns: int, nt: int, require_1to1: bool = True) -> None:
    if y.ndim != 2 or y.size(0) != 2:
        raise ValueError(f"y must have shape [2,K], got {tuple(y.shape)}")

    if y.numel() == 0:
        return

    src = y[0].tolist()
    tgt = y[1].tolist()

    if any(i < 0 or i >= ns for i in src):
        raise ValueError(f"y src index out of range (ns={ns}): {src[:20]}")
    if any(j < 0 or j >= nt for j in tgt):
        raise ValueError(f"y tgt index out of range (nt={nt}): {tgt[:20]}")

    if require_1to1:
        if len(src) != len(set(src)):
            raise ValueError("y is not functional (duplicate source indices).")
        if len(tgt) != len(set(tgt)):
            raise ValueError("y is not injective (duplicate target indices).")


def maybe_swap_source_target(
    g_a: Dict[str, Any],
    g_b: Dict[str, Any],
    y: Optional[torch.Tensor],
    enforce_source_le_target: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[torch.Tensor], bool]:
    """
    Erzwingt w.l.o.g. |V_s| <= |V_t| (DGMC-Paper-typische Annahme).
    Wenn geswapped wird, wird y entsprechend getauscht: [src,tgt] -> [tgt,src].
    """
    na = _num_nodes(g_a)
    nb = _num_nodes(g_b)

    if enforce_source_le_target and na > nb:
        if y is not None:
            y = torch.stack([y[1], y[0]], dim=0)
        return g_b, g_a, y, True
    return g_a, g_b, y, False


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class DGMCPairJsonlDataset(Dataset):
    """
    Lädt JSONL-Pairs und konvertiert graph_a/graph_b → PyG Data (nur Tensorfelder),
    sowie y → LongTensor [2,K].

    Wichtig: y-Indizes beziehen sich auf die Knotenreihenfolge im JSON (nodes-Liste),
    und graph_pipeline.json_graph_to_pyg muss dieselbe Reihenfolge beibehalten.
    """

    def __init__(
        self,
        pairs_path: str | Path,
        max_pairs: Optional[int] = None,
        use_only_positive: bool = True,
        undirected: bool = True,
        enforce_source_le_target: bool = True,
        require_1to1: bool = True,
        require_full_source_coverage: bool = False,
        allow_pairs_without_y: bool = False,
    ) -> None:
        super().__init__()
        self.pairs_path = Path(pairs_path)
        self.max_pairs = max_pairs
        self.use_only_positive = use_only_positive
        self.undirected = undirected
        self.enforce_source_le_target = enforce_source_le_target
        self.require_1to1 = require_1to1
        self.require_full_source_coverage = require_full_source_coverage
        self.allow_pairs_without_y = allow_pairs_without_y

        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_path}")

        self._pairs: List[Dict[str, Any]] = []
        for _, obj in _iter_jsonl(self.pairs_path, max_lines=max_pairs):
            label = int(obj.get("label", 1))
            if self.use_only_positive and label != 1:
                continue
            # In manchen Settings willst du Paare ohne y trotzdem laden (z.B. Eval)
            if label == 1 and (obj.get("y") is None) and not self.allow_pairs_without_y:
                # skip silently? -> lieber hart:
                raise ValueError("Positive pair without y encountered (label=1, y=None).")
            self._pairs.append(obj)

        if not self._pairs:
            raise ValueError(f"No pairs loaded from {self.pairs_path} (use_only_positive={use_only_positive}).")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> PairItem:
        p = self._pairs[idx]

        g_a = p.get("graph_a")
        g_b = p.get("graph_b")
        if g_a is None or g_b is None:
            raise KeyError("Pair must contain graph_a and graph_b keys.")

        label = int(p.get("label", 1))
        y = parse_y(p.get("y", None))

        # optional swap to satisfy |V_s| <= |V_t|
        g_s, g_t, y, swapped = maybe_swap_source_target(
            g_a, g_b, y, enforce_source_le_target=self.enforce_source_le_target
        )

        # JSON → PyG (can include meta fields), then reduce to tensor-only Data
        d_s_full = gp.json_graph_to_pyg(g_s, undirected=self.undirected)
        d_t_full = gp.json_graph_to_pyg(g_t, undirected=self.undirected)

        # Tensor-only to avoid PyG collation issues with dict/list fields
        d_s = Data(
            x=d_s_full.x,
            edge_index=d_s_full.edge_index,
            edge_attr=getattr(d_s_full, "edge_attr", None),
        )
        d_t = Data(
            x=d_t_full.x,
            edge_index=d_t_full.edge_index,
            edge_attr=getattr(d_t_full, "edge_attr", None),
        )

        ns = int(d_s.num_nodes)
        nt = int(d_t.num_nodes)

        if label == 1 and y is not None:
            validate_y(y, ns, nt, require_1to1=self.require_1to1)

            if self.require_full_source_coverage and int(y.size(1)) != ns:
                raise ValueError(
                    f"Pair violates full-source coverage: |y|={int(y.size(1))} != ns={ns}."
                )

        meta = {
            "graph_id_s": g_s.get("graph_id"),
            "graph_id_t": g_t.get("graph_id"),
            "ns": ns,
            "nt": nt,
            "aug": p.get("aug"),
        }

        return PairItem(
            data_s=d_s,
            data_t=d_t,
            y=y,
            label=label,
            swapped=swapped,
            meta=meta,
        )

    def describe(self) -> Dict[str, Any]:
        """
        Kleine Statistik für Thesis/Debug: Swap-Rate, Größenbereiche, y-Größen.
        """
        total = len(self)
        swapped = 0
        ns_list: List[int] = []
        nt_list: List[int] = []
        y_sizes: List[int] = []
        labels = []

        full_source = 0  # Anzahl Paare mit yK == ns
        pairs_with_y = 0  # Anzahl Paare, die überhaupt y haben

        for i in range(total):
            it = self[i]
            labels.append(it.label)
            ns = int(it.meta["ns"])
            nt = int(it.meta["nt"])
            ns_list.append(ns)
            nt_list.append(nt)
            if it.swapped:
                swapped += 1
            if it.y is not None:
                k = int(it.y.size(1))
                y_sizes.append(k)
                pairs_with_y += 1
                if k == ns:
                    full_source += 1

        def q(xs: Sequence[int], frac: float) -> int:
            if not xs:
                return 0
            s = sorted(xs)
            pos = int(round(frac * (len(s) - 1)))
            return int(s[pos])

        return {
            "pairs_path": str(self.pairs_path),
            "pairs_loaded": total,
            "labels": {"pos": int(sum(1 for x in labels if x == 1)), "neg": int(sum(1 for x in labels if x == 0))},
            "use_only_positive": self.use_only_positive,
            "enforce_source_le_target": self.enforce_source_le_target,
            "undirected": self.undirected,
            "swap_rate_pct": round(100.0 * swapped / total, 2) if total else 0.0,
            "ns": {"min": min(ns_list), "q50": q(ns_list, 0.5), "max": max(ns_list)},
            "nt": {"min": min(nt_list), "q50": q(nt_list, 0.5), "max": max(nt_list)},
            "yK": {"min": min(y_sizes) if y_sizes else 0, "q50": q(y_sizes, 0.5), "max": max(y_sizes) if y_sizes else 0},
            "rate_full_source": {
                "count": full_source,
                "total_with_y": pairs_with_y,
                "pct": round(100.0 * full_source / pairs_with_y, 2) if pairs_with_y else 0.0,
            },
        }


# ---------------------------------------------------------------------
# Collate for DGMC batching
# ---------------------------------------------------------------------
def collate_pairs(batch: Sequence[PairItem]) -> Dict[str, torch.Tensor]:
    """
    Baut Batch-Tensoren passend zur DGMC.forward-Signatur.

    Rückgabe:
      x_s, edge_index_s, edge_attr_s, batch_s,
      x_t, edge_index_t, edge_attr_t, batch_t,
      y (LongTensor [2, sum K])

    Wichtig:
    - y[0] wird um den kumulativen Source-Offset verschoben.
    - y[1] bleibt lokaler Target-Index (DGMC spaltenweise pro Graph gepaddet).
    """
    if not batch:
        raise ValueError("Empty batch")

    # Batch graphs with PyG (safe because Data contains only tensors)
    bs: Batch = Batch.from_data_list([it.data_s for it in batch])
    bt: Batch = Batch.from_data_list([it.data_t for it in batch])

    # Build y for the batch
    y_rows: List[torch.Tensor] = []
    y_cols: List[torch.Tensor] = []

    # compute per-graph source offsets in the *flattened* source node list
    s_offset = 0
    for it in batch:
        ns = int(it.data_s.num_nodes)
        if it.y is None:
            raise ValueError("collate_pairs expects y for all items (use_only_positive=True).")
        y_rows.append(it.y[0] + s_offset)
        y_cols.append(it.y[1])  # keep local target indices
        s_offset += ns

    y = torch.cat([torch.stack([r, c], dim=0) for r, c in zip(y_rows, y_cols)], dim=1)

    return {
        "x_s": bs.x,
        "edge_index_s": bs.edge_index,
        "edge_attr_s": getattr(bs, "edge_attr", None),
        "batch_s": bs.batch,
        "x_t": bt.x,
        "edge_index_t": bt.edge_index,
        "edge_attr_t": getattr(bt, "edge_attr", None),
        "batch_t": bt.batch,
        "y": y,
    }


# ---------------------------------------------------------------------
# Optional quick self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal sanity run when executing this file directly.
    pairs = Path("data/synthetic_training_pairs.jsonl")
    if not pairs.exists():
        print(f"[dgmc_dataset] Hinweis: Default-Pfad existiert nicht: {pairs}")
        sys.exit(0)

    ds = DGMCPairJsonlDataset(
        pairs_path=pairs,
        use_only_positive=True,
        undirected=True,
        enforce_source_le_target=True,
        require_1to1=True,
        allow_pairs_without_y=True,
    )
    print(json.dumps(ds.describe(), indent=2, ensure_ascii=False))

    # try collate first 4 positives
    positives = [ds[i] for i in range(len(ds)) if ds[i].label == 1][:4]
    batch = collate_pairs(positives)
    print("[dgmc_dataset] collate ok:", {k: (None if v is None else tuple(v.shape)) for k, v in batch.items()})
