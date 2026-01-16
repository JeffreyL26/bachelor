from __future__ import annotations

"""
dgmc_matcher.py

Match real *Ist*-graphs against LBS templates using a trained DGMC model.

This script is designed to be **compatible with your current stack**:
  - graph_pipeline.py (current): encodes node features as
      [node_type_onehot (4)] + [direction_onehot incl. unknown (4)]  => 8 dims
    and edge features as one-hot of (MEMA, METR, MENE, MEME, unknown) => 5 dims.
  - dgmc_template_training.py (current): trains DGMC on synthetic pairs with supervision `y`.

What you get
------------
For each Ist-graph (from a JSONL file), the script:
  1) scores the Ist-graph against every template (graph-level heuristic from DGMC S)
  2) outputs the **Top-N templates** (default N=3)
  3) for the best template, outputs **Top-K node candidates per template node** (default K=3)

Evaluation (labelled subset)
----------------------------
To make DGMC outputs comparable to the rule/constraints and GED baselines, this
matcher can evaluate on your labelled subset derived from `BNDL2MC.csv`.

- Default: tries to load `data/training_data/BNDL2MC.csv` (set `--bndl2mc_path ''` to disable).
- Evaluation uses (MaLo, MeLo) pairs only AFTER inference; it does not affect matching.

At the end it prints:
  - Top-1 prediction distribution over all processed Ist graphs
  - Top-1 / Top-3 accuracy on the labelled subset (where a template label can be derived)

Important notes (re: ID leakage)
--------------------------------
- graph_pipeline.py uses node["id"] only to map edges to indices; it is NOT part of x.
  => DGMC cannot "cheat" by matching IDs.

How scoring works
-----------------
DGMC produces a similarity matrix S with shape [n_source, n_target].
Default score = mean over source-nodes of max similarity in that row:
  score = mean_i max_j S[i, j]

Optional (recommended when |V_template| and |V_ist| differ a lot):
  --score_mode mean_rowmax_symmetric
which averages both directions:
  0.5 * (score(template->ist) + score(ist->template))

Run
---
Example:
  python dgmc_matcher.py \
    --ist_path data/ist_graphs_all.jsonl \
    --templates_path data/lbs_soll_graphs.jsonl \
    --checkpoint data/dgmc_partial.pt \
    --out_path runs/dgmc_matches_top3.jsonl \
    --top_templates 3 \
    --top_matches 3
"""

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# --- DGMC robust import (same pattern as training) ---
try:
    from dgmc.models import DGMC  # type: ignore
except Exception:  # pragma: no cover
    from torch_geometric.nn import DGMC  # type: ignore

from torch_geometric.nn import GINEConv

import graph_pipeline as gp


# =============================================================================
# Model definition (MUST match dgmc_template_training.py)
# =============================================================================


class EdgeAwareGINE(nn.Module):
    """Small GINE-style GNN that supports edge_attr.

    Copied (verbatim in logic) from your current dgmc_template_training.py
    to avoid import coupling.

    DGMC expects psi_2 to expose `in_channels` and `out_channels` attributes.
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
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))

        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin_out(x)


# =============================================================================
# IO helpers
# =============================================================================

TGraph = Dict[str, Any]


def _resolve_default(path: Path) -> Path:
    """Try a couple of sensible fallbacks (similar to your baselines)."""
    if path.exists():
        return path
    alt = Path(__file__).resolve().parent / path.name
    if alt.exists():
        return alt
    return path


def iter_jsonl(path: Path, max_lines: Optional[int] = None) -> Iterable[TGraph]:
    """Stream JSONL (optionally truncated)."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_lines is not None and line_no > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e


def load_jsonl(path: Path, max_lines: Optional[int] = None) -> List[TGraph]:
    return list(iter_jsonl(path, max_lines=max_lines))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Optional: label mapping from BNDL2MC.csv (evaluation only)
# =============================================================================

MCID_TO_TEMPLATE_LABEL: Dict[str, str] = {
    # Mapping used across your project so far.
    # Extend if you add more labelled MCIDs.
    "S_A1_A2": "9992000000042",
    "S_C3": "9992000000175",
    "S_A001": "9992000000026",
}


def load_bndl2mc_labels(bndl2mc_path: Path) -> Dict[Tuple[str, str], str]:
    """Return map (malo_id, melo_id) -> MCID.

    Expects semicolon-separated CSV with columns:
      Bündel;Marktlokation;Messlokation;MCID
    """
    mapping: Dict[Tuple[str, str], str] = {}
    with bndl2mc_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(";")
        cols = {name: i for i, name in enumerate(header)}
        need = {"Marktlokation", "Messlokation", "MCID"}
        if not need.issubset(cols.keys()):
            raise ValueError(
                f"BNDL2MC header missing required columns {sorted(need)}. "
                f"Got: {header}"
            )

        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) <= max(cols.values()):
                continue

            # MaLo sometimes stored numeric => normalise by int-cast if possible
            try:
                malo = str(int(parts[cols["Marktlokation"]]))
            except Exception:
                malo = str(parts[cols["Marktlokation"]]).strip()

            melo = str(parts[cols["Messlokation"]]).strip()
            mcid = str(parts[cols["MCID"]]).strip()

            if not malo or not melo or not mcid:
                continue

            mapping[(malo, melo)] = mcid

    return mapping


def infer_ground_truth_for_ist(
    g: TGraph,
    pair_to_mcid: Dict[Tuple[str, str], str],
) -> Optional[Dict[str, str]]:
    """If any (MaLo,MeLo) pair of the graph occurs in BNDL2MC, return ground truth.

    Note: if the MCID has no known mapping to a template label, template_label is "".
    """
    nodes = g.get("nodes") or []
    malos = [
        str(n.get("id"))
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "MaLo" and n.get("id") is not None
    ]
    melos = [
        str(n.get("id"))
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "MeLo" and n.get("id") is not None
    ]

    found: Optional[Tuple[str, str]] = None
    found_mcid: Optional[str] = None
    for malo in malos:
        for melo in melos:
            mcid = pair_to_mcid.get((malo, melo))
            if mcid:
                found = (malo, melo)
                found_mcid = mcid
                break
        if found_mcid:
            break

    if not found_mcid:
        return None

    return {
        "mcid": found_mcid,
        "template_label": MCID_TO_TEMPLATE_LABEL.get(found_mcid, ""),
        "malo": found[0] if found else "",
        "melo": found[1] if found else "",
    }


# =============================================================================
# Checkpoint loading / model rebuild
# =============================================================================


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(path, map_location=device)

    # Expected from dgmc_template_training.py:
    # {"epoch": ..., "model_state": ..., "optimizer_state": ..., "args": ..., "in_channels": ..., "edge_dim": ...}
    if isinstance(obj, dict) and "model_state" in obj:
        return obj

    # Accept plain state_dict as a fallback
    if isinstance(obj, dict):
        return {"model_state": obj, "args": {}, "in_channels": None, "edge_dim": None}

    raise ValueError(
        "Unsupported checkpoint format. Expected a dict with key 'model_state' "
        "(from dgmc_template_training.py) or a plain state_dict dict."
    )


def _infer_dims_from_graph(sample_graph: TGraph, undirected: bool) -> Tuple[int, int]:
    d = gp.json_graph_to_pyg(sample_graph, undirected=undirected)
    in_channels = int(d.x.size(-1))
    edge_attr = getattr(d, "edge_attr", None)
    if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.size(1) > 0:
        edge_dim = int(edge_attr.size(1))
    else:
        # graph_pipeline.py uses 4 known + 1 unknown => 5
        edge_dim = 5
    return in_channels, edge_dim


def build_model_from_checkpoint(
    ckpt: Dict[str, Any],
    *,
    sample_graph: TGraph,
    undirected: bool,
    device: torch.device,
    override_num_steps: Optional[int] = None,
    override_k: Optional[int] = None,
    override_detach: Optional[bool] = None,
) -> DGMC:
    ckpt_args: Dict[str, Any] = dict(ckpt.get("args") or {})

    # Dimensions
    in_channels = ckpt.get("in_channels")
    edge_dim = ckpt.get("edge_dim")
    if (
        not isinstance(in_channels, int)
        or not isinstance(edge_dim, int)
        or in_channels <= 0
        or edge_dim <= 0
    ):
        in_channels, edge_dim = _infer_dims_from_graph(sample_graph, undirected=undirected)

    # Architecture params
    hidden_channels = int(ckpt_args.get("hidden_channels", 64))
    out_channels = int(ckpt_args.get("out_channels", 64))
    num_layers = int(ckpt_args.get("num_layers", 3))
    dropout = float(ckpt_args.get("dropout", 0.0))

    num_steps = int(ckpt_args.get("num_steps", 10))
    k = int(ckpt_args.get("k", -1))
    detach = bool(ckpt_args.get("detach", False))

    if override_num_steps is not None:
        num_steps = int(override_num_steps)
    if override_k is not None:
        k = int(override_k)
    if override_detach is not None:
        detach = bool(override_detach)

    psi_1 = EdgeAwareGINE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=edge_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    psi_2 = EdgeAwareGINE(
        in_channels=out_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=edge_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    model = DGMC(
        psi_1=psi_1,
        psi_2=psi_2,
        num_steps=num_steps,
        k=k,
        detach=detach,
    ).to(device)

    state_dict = ckpt["model_state"]
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint['model_state'] must be a state_dict dict")

    # Be strict by default: you want to know if architectures diverged.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# =============================================================================
# Matching logic
# =============================================================================


@dataclass
class PreparedGraph:
    graph_json: TGraph
    data: Any  # PyG Data
    node_ids: List[Any]
    node_types: List[Any]


def prepare_graph(g: TGraph, *, undirected: bool, device: torch.device) -> PreparedGraph:
    d = gp.json_graph_to_pyg(g, undirected=undirected).to(device)
    node_ids = list(getattr(d, "node_ids", []))
    node_types = list(getattr(d, "node_types", []))
    return PreparedGraph(graph_json=g, data=d, node_ids=node_ids, node_types=node_types)


@torch.no_grad()
def dgmc_similarity_matrix(
    model: DGMC,
    *,
    src: PreparedGraph,
    tgt: PreparedGraph,
    device: torch.device,
) -> torch.Tensor:
    """Compute dense similarity S [n_src, n_tgt] for one graph pair."""

    x_s = src.data.x
    ei_s = src.data.edge_index
    ea_s = getattr(src.data, "edge_attr", None)

    x_t = tgt.data.x
    ei_t = tgt.data.edge_index
    ea_t = getattr(tgt.data, "edge_attr", None)

    ns = int(x_s.size(0))
    nt = int(x_t.size(0))

    batch_s = torch.zeros(ns, dtype=torch.long, device=device)
    batch_t = torch.zeros(nt, dtype=torch.long, device=device)

    S0, SL = model(x_s, ei_s, ea_s, batch_s, x_t, ei_t, ea_t, batch_t, y=None)

    # Prefer the refined score if refinement is enabled
    S = SL if SL is not None else S0

    # DGMC may return a sparse-ish container in some modes
    if hasattr(S, "to_dense"):
        S = S.to_dense()

    # Safety: if padded, cut to true nt
    if isinstance(S, torch.Tensor) and S.dim() == 2 and S.size(1) > nt:
        S = S[:, :nt]

    if not isinstance(S, torch.Tensor) or S.dim() != 2:
        raise RuntimeError("DGMC produced an unexpected score matrix type/shape")

    return S


def score_mean_rowmax(S: torch.Tensor) -> float:
    if S.numel() == 0:
        return float("-inf")
    return float(S.max(dim=1).values.mean().item())


@torch.no_grad()
def score_pair(
    model: DGMC,
    *,
    template: PreparedGraph,
    ist: PreparedGraph,
    device: torch.device,
    score_mode: str,
) -> Tuple[float, torch.Tensor]:
    """Return (score, S_template_to_ist).

    We always return S(template->ist), because we use it for node-level Top-K output.
    """
    S_t2i = dgmc_similarity_matrix(model, src=template, tgt=ist, device=device)
    s1 = score_mean_rowmax(S_t2i)

    if score_mode == "mean_rowmax":
        return s1, S_t2i

    if score_mode == "mean_rowmax_symmetric":
        S_i2t = dgmc_similarity_matrix(model, src=ist, tgt=template, device=device)
        s2 = score_mean_rowmax(S_i2t)
        return 0.5 * (s1 + s2), S_t2i

    raise ValueError(f"Unknown score_mode: {score_mode}")


def topk_from_S(
    S: torch.Tensor,
    *,
    src: PreparedGraph,
    tgt: PreparedGraph,
    k_top: int,
) -> List[Dict[str, Any]]:
    """For each src node, return top-k target candidates."""
    if S.dim() != 2:
        return []

    ns, nt = int(S.size(0)), int(S.size(1))
    if ns <= 0:
        return []

    k_eff = int(min(max(int(k_top), 1), nt)) if nt > 0 else 0

    out: List[Dict[str, Any]] = []
    if k_eff == 0:
        for i in range(ns):
            out.append(
                {
                    "src_index": i,
                    "src_node_id": src.node_ids[i] if i < len(src.node_ids) else None,
                    "src_type": src.node_types[i] if i < len(src.node_types) else None,
                    "candidates": [],
                }
            )
        return out

    vals, idxs = torch.topk(S, k=k_eff, dim=1)
    vals_cpu = vals.detach().cpu()
    idxs_cpu = idxs.detach().cpu()

    for i in range(ns):
        cand_list = []
        for r in range(k_eff):
            j = int(idxs_cpu[i, r].item())
            cand_list.append(
                {
                    "tgt_index": j,
                    "tgt_node_id": tgt.node_ids[j] if j < len(tgt.node_ids) else None,
                    "tgt_type": tgt.node_types[j] if j < len(tgt.node_types) else None,
                    "score": float(vals_cpu[i, r].item()),
                }
            )
        out.append(
            {
                "src_index": i,
                "src_node_id": src.node_ids[i] if i < len(src.node_ids) else None,
                "src_type": src.node_types[i] if i < len(src.node_types) else None,
                "candidates": cand_list,
            }
        )

    return out


def template_label(g: TGraph) -> Optional[str]:
    ga = g.get("graph_attrs", {}) or {}
    x = ga.get("lbs_code") or g.get("label") or g.get("graph_id")
    return str(x) if x is not None else None


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match Ist graphs to templates with a trained DGMC model.")

    p.add_argument("--ist_path", type=str, default=os.path.join("data", "ist_graphs_all.jsonl"))
    p.add_argument("--templates_path", type=str, default=os.path.join("data", "lbs_soll_graphs.jsonl"))
    p.add_argument("--checkpoint", type=str, default=os.path.join("data", "dgmc_partial.pt"))
    p.add_argument("--out_path", type=str, default=os.path.join("runs", "dgmc_matches_top3.jsonl"))

    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--undirected", action="store_true", help="Use undirected edges (should match training).")
    p.add_argument("--directed", action="store_true", help="Use directed edges (overrides --undirected).")

    p.add_argument("--top_templates", type=int, default=3, help="How many best templates to return per Ist graph.")
    p.add_argument("--top_matches", type=int, default=3, help="Top-k node candidates per template node (best template).")

    p.add_argument(
        "--score_mode",
        type=str,
        default="mean_rowmax",
        choices=["mean_rowmax", "mean_rowmax_symmetric"],
        help="Graph-level ranking heuristic derived from DGMC similarity matrix.",
    )

    # Debug: truncation
    p.add_argument("--max_ist", type=int, default=0, help="Process only the first N ist graphs (0 = all).")
    p.add_argument("--max_templates", type=int, default=0, help="Use only the first N templates (0 = all).")

    # Optional: DGMC inference overrides (should usually match training)
    p.add_argument("--override_num_steps", type=int, default=-999)
    p.add_argument("--override_k", type=int, default=-999)
    p.add_argument("--override_detach", action="store_true")
    p.add_argument("--no_override_detach", action="store_true")

    # Evaluation (default ON, matching your baseline scripts)
    p.add_argument(
        "--bndl2mc_path",
        type=str,
        default=str(os.path.join("data", "training_data", "BNDL2MC.csv")),
        help="BNDL2MC.csv for evaluation (default: data/training_data/BNDL2MC.csv). Use '' to disable.",
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    # directed/undirected flag resolution (keep your current default: undirected)
    undirected = True
    if args.directed:
        undirected = False
    elif args.undirected:
        undirected = True

    device = torch.device(args.device)

    ist_path = _resolve_default(Path(args.ist_path))
    tpl_path = _resolve_default(Path(args.templates_path))
    ckpt_path = _resolve_default(Path(args.checkpoint))
    out_path = Path(args.out_path)

    if not ist_path.exists():
        raise FileNotFoundError(f"Ist-Graph JSONL not found: {ist_path}")
    if not tpl_path.exists():
        raise FileNotFoundError(f"Template JSONL not found: {tpl_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DGMC checkpoint not found: {ckpt_path}")

    ensure_dir(out_path.parent)

    max_ist = None if int(args.max_ist) <= 0 else int(args.max_ist)
    max_templates = None if int(args.max_templates) <= 0 else int(args.max_templates)

    if max_ist is not None:
        print(f"[dgmc][note] --max_ist={max_ist} => the Ist file will NOT be processed completely.")
    if max_templates is not None:
        print(f"[dgmc][note] --max_templates={max_templates} => the Templates file will NOT be processed completely.")

    templates = load_jsonl(tpl_path, max_lines=max_templates)
    if not templates:
        raise RuntimeError("No templates loaded (templates JSONL empty or invalid).")

    # --- Evaluation data (optional) ---
    pair_to_mcid: Dict[Tuple[str, str], str] = {}
    bndl_arg = str(args.bndl2mc_path or "").strip()
    if bndl_arg:
        bndl_path = _resolve_default(Path(bndl_arg))
        if bndl_path.exists():
            pair_to_mcid = load_bndl2mc_labels(bndl_path)
            print(f"[dgmc] Loaded BNDL2MC pairs: {len(pair_to_mcid)} from {bndl_path}")
        else:
            print(f"[dgmc][warn] BNDL2MC.csv not found at: {bndl_path} (evaluation disabled)")

    # Build + load model
    ckpt = _load_checkpoint(ckpt_path, device=device)

    override_num_steps = None if args.override_num_steps == -999 else int(args.override_num_steps)
    override_k = None if args.override_k == -999 else int(args.override_k)
    override_detach: Optional[bool] = None
    if args.override_detach and args.no_override_detach:
        raise ValueError("Use only one of --override_detach / --no_override_detach")
    if args.override_detach:
        override_detach = True
    if args.no_override_detach:
        override_detach = False

    model = build_model_from_checkpoint(
        ckpt,
        sample_graph=templates[0],
        undirected=undirected,
        device=device,
        override_num_steps=override_num_steps,
        override_k=override_k,
        override_detach=override_detach,
    )

    # Pre-prepare templates once
    prepared_templates: List[PreparedGraph] = [
        prepare_graph(t, undirected=undirected, device=device) for t in templates
    ]

    # Evaluation counters
    eval_total = 0
    eval_top1 = 0
    eval_top3 = 0

    # Prediction distribution
    pred_top1_dist: Counter = Counter()

    # Stream ist graphs (avoid loading everything into RAM)
    written = 0
    ist_iter = iter_jsonl(ist_path, max_lines=max_ist)

    print(
        f"[dgmc] device={device} | undirected={undirected} | ist_graphs={'ALL' if max_ist is None else max_ist} "
        f"| templates={len(prepared_templates)} | top_k={args.top_templates} | top_matches={args.top_matches} "
        f"| score_mode={args.score_mode}"
    )

    with out_path.open("w", encoding="utf-8") as f_out:
        for idx, g_ist in enumerate(ist_iter, start=1):
            ist_p = prepare_graph(g_ist, undirected=undirected, device=device)

            tpl_scores: List[Tuple[float, int]] = []

            best_i = -1
            best_score = float("-inf")
            best_S_t2i: Optional[torch.Tensor] = None

            for ti, tpl_p in enumerate(prepared_templates):
                sc, S_t2i = score_pair(
                    model,
                    template=tpl_p,
                    ist=ist_p,
                    device=device,
                    score_mode=args.score_mode,
                )

                tpl_scores.append((float(sc), ti))
                if sc > best_score:
                    best_score = float(sc)
                    best_i = ti
                    best_S_t2i = S_t2i

            tpl_scores.sort(key=lambda x: (-x[0], x[1]))
            top_n = max(1, int(args.top_templates))
            top_tpl = tpl_scores[:top_n]

            top_templates_out = []
            for rank, (sc, ti) in enumerate(top_tpl, start=1):
                t = templates[ti]
                top_templates_out.append(
                    {
                        "rank": rank,
                        "template_graph_id": t.get("graph_id"),
                        "template_label": template_label(t),
                        "score": float(sc),
                    }
                )

            # Distribution over top-1 predictions
            if top_templates_out:
                pred_top1_dist[str(top_templates_out[0].get("template_label") or "")] += 1

            best_tpl = templates[best_i]
            best_tpl_p = prepared_templates[best_i]

            # Per-template-node Top-K candidates (best template only)
            if best_S_t2i is None:
                topk = []
            else:
                topk = topk_from_S(
                    best_S_t2i, src=best_tpl_p, tgt=ist_p, k_top=int(args.top_matches)
                )

            # Ground truth (evaluation only; does NOT affect inference)
            gt = infer_ground_truth_for_ist(g_ist, pair_to_mcid) if pair_to_mcid else None
            if gt and gt.get("template_label"):
                eval_total += 1
                gt_label = str(gt["template_label"])
                pred1 = str(top_templates_out[0].get("template_label") or "") if top_templates_out else ""
                if pred1 == gt_label:
                    eval_top1 += 1
                pred_labels = [str(x.get("template_label") or "") for x in top_templates_out[:3]]
                if gt_label in pred_labels:
                    eval_top3 += 1

            res = {
                "ist_graph_id": g_ist.get("graph_id"),
                "top_templates": top_templates_out,
                "ground_truth": gt,
                # DGMC-specific extras (kept for debugging/analysis):
                "best_template_graph_id": best_tpl.get("graph_id"),
                "best_template_label": template_label(best_tpl),
                "best_score": float(best_score),
                "topk": topk,
            }

            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            written += 1

            if idx % 50 == 0:
                print(f"[dgmc] processed {idx} ist graphs")

    print(f"[dgmc] wrote: {out_path}")

    # Report prediction distribution (like your WL/GED logs)
    total_preds = sum(int(v) for v in pred_top1_dist.values())
    if total_preds > 0:
        print("[dgmc] top1 prediction distribution (processed Ist graphs):")
        for lab, cnt in pred_top1_dist.most_common():
            pct = 100.0 * float(cnt) / float(total_preds)
            print(f"  - {lab}: {cnt} ({pct:.3f}%)")

    # Report evaluation
    if eval_total > 0:
        print(
            "[dgmc] evaluation on labelled subset | "
            f"n={eval_total} | top1={eval_top1/eval_total:.3f} | top3={eval_top3/eval_total:.3f}"
        )
    else:
        if pair_to_mcid:
            print(
                "[dgmc][warn] evaluation skipped: no labelled graphs found that map to known template labels. "
                "If you added more MCIDs, extend MCID_TO_TEMPLATE_LABEL."
            )
        else:
            print("[dgmc] evaluation skipped (no BNDL2MC loaded).")

    print(f"[dgmc] done. wrote {written} results")


if __name__ == "__main__":
    main()
