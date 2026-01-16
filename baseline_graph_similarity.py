"""
baseline_graph_edit_distance.py

Graph similarity baseline based on (exact) Graph Edit Distance (GED).

Why this baseline exists
-----------------------
Your learned DGMC matcher exploits node/edge features and learns a similarity
function. For the thesis you also need a *transparent* non-learned baseline that
compares each Ist-graph against each template graph and returns Top-k matches.

This file implements GED using **NetworkX**'s exact `graph_edit_distance`
algorithm (Abu-Aisheh et al.). Exact GED is NP-hard in general, but in your
project setting graphs are typically small enough to make it a practical
baseline (15 templates; Ist graphs with few nodes/edges).

Important design choices
------------------------
- Works directly with your JSONL graphs: `data/ist_graphs_all.jsonl` and
  `data/lbs_soll_graphs.jsonl`.
- Node labels: (type, direction). Direction extraction is **pipeline-conform**
  by reusing helpers from `baseline_constraints.py` where possible.
- Template directions: many template JSON nodes do not carry direction. If
  `--lbs_json_dir` is provided (default: `data/lbs_templates`), the baseline
  enriches template node directions from `_lbs_optionality.lbs_objects` using
  the same robust key logic as the rule baseline.
- Edge labels: relation type (`rel`, e.g., MEMA/METR/MENE).
- Similarity score in [0,1]:
      score = 1 - GED / (|V1|+|V2|+|E1|+|E2|)
  (with unit costs). Lower GED => higher score.

Evaluation (optional, but default ON)
-------------------------------------
If `--bndl2mc_path` points to an existing CSV, the script derives ground-truth
labels for a small subset of Ist graphs (via MaLo/MeLo pairs) and prints Top-1
and Top-3 accuracy. This is identical in spirit to your rule baseline:
the labels are *not* used for matching, only for evaluation.

Run
---
  python baseline_graph_edit_distance.py \
      --ist_path data/ist_graphs_all.jsonl \
      --templates_path data/lbs_soll_graphs.jsonl \
      --lbs_json_dir data/lbs_templates \
      --out_path data/ged_baseline_matches.jsonl \
      --top_k 3

Notes on feasibility / runtime
------------------------------
Exact GED can blow up for larger graphs. If you ever face a pathological case,
use `--timeout_s` to cap per-pair runtime. This trades exactness for robustness.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance


JsonGraph = Dict[str, Any]


# ---------------------------------------------------------------------------
# Reuse project helpers where available (to avoid redundancy)
# ---------------------------------------------------------------------------

# We intentionally depend on your existing baseline_constraints.py for:
# - JSONL loading helpers
# - pipeline-conform direction extraction
# - (optional) BNDL2MC evaluation helpers
#
# If the import fails (e.g., when someone copies this file standalone), we fall
# back to small local implementations.
try:
    from baseline_constraints import (  # type: ignore
        _ensure_dir,
        _iter_jsonl,
        _load_jsonl,
        _node_direction,
        _resolve_default,
        infer_ground_truth_for_ist,
        load_bndl2mc_labels,
        load_lbs_optionality_catalog,
        _lbs_object_direction,
    )
except Exception:  # pragma: no cover
    BASE_DIR = Path(__file__).resolve().parent

    def _resolve_default(path: Path) -> Path:
        if path.exists():
            return path
        alt = BASE_DIR / path.name
        if alt.exists():
            return alt
        return path

    def _ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def _iter_jsonl(path: Path, max_lines: Optional[int] = None) -> Iterable[JsonGraph]:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if max_lines is not None and i > max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _load_jsonl(path: Path, max_lines: Optional[int] = None) -> List[JsonGraph]:
        return list(_iter_jsonl(path, max_lines=max_lines))

    # Fallback direction extraction: keep it minimal (unknown everywhere).
    def _node_direction(node: Dict[str, Any]) -> str:
        _ = node
        return "unknown"

    # Evaluation helpers disabled in fallback mode.
    def load_bndl2mc_labels(_: Path) -> Dict[Tuple[str, str], str]:
        return {}

    def infer_ground_truth_for_ist(_: JsonGraph, __: Dict[Tuple[str, str], str]) -> Optional[Dict[str, str]]:
        return None

    def load_lbs_optionality_catalog(_: Path) -> Dict[str, Dict[str, Any]]:
        return {}

    def _lbs_object_direction(_: Dict[str, Any]) -> str:
        return "unknown"


# ---------------------------------------------------------------------------
# NetworkX graph construction
# ---------------------------------------------------------------------------


def _edge_rel(e: Dict[str, Any]) -> str:
    """Extract and normalise relation label for an edge dict."""
    rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
    if rel is None:
        return "UNKNOWN"
    s = str(rel).strip()
    return s.upper() if s else "UNKNOWN"


def json_to_nx_graph(
    g: JsonGraph,
    *,
    directed: bool = False,
    enrich_template_dirs: Optional[Dict[str, str]] = None,
) -> nx.Graph:
    """Convert a JSON graph into a NetworkX (Di)Graph.

    Parameters
    ----------
    directed:
        If False (default), build an undirected Graph.
    enrich_template_dirs:
        Optional mapping object_code -> direction for template nodes.
        Only used if a node's direction is "unknown".
    """
    G: nx.Graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    nodes = [n for n in (g.get("nodes") or []) if isinstance(n, dict)]
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        node_id = str(nid)
        ntype = str(n.get("type") or "UNKNOWN")
        attrs = n.get("attrs") if isinstance(n.get("attrs"), dict) else {}
        obj_code = None
        if isinstance(attrs, dict):
            obj_code = attrs.get("object_code")

        d = _node_direction(n)
        if d == "unknown" and enrich_template_dirs and obj_code is not None:
            d2 = enrich_template_dirs.get(str(obj_code))
            if d2:
                d = str(d2)

        # Keep only attributes that matter for matching (avoid noise).
        G.add_node(node_id, type=ntype, direction=d)

    edges = [e for e in (g.get("edges") or []) if isinstance(e, dict)]
    for e in edges:
        s = e.get("src")
        t = e.get("dst")
        if s is None or t is None:
            continue
        u = str(s)
        v = str(t)
        if not (G.has_node(u) and G.has_node(v)):
            # Robustness: ignore dangling edges.
            continue
        rel = _edge_rel(e)

        # If an undirected graph has duplicate edges from both directions, store
        # them as a small set (single edge with a tuple label).
        if not directed and G.has_edge(u, v):
            prev = G[u][v].get("rel")
            if isinstance(prev, tuple):
                rels = set(prev)
                rels.add(rel)
                G[u][v]["rel"] = tuple(sorted(rels))
            elif isinstance(prev, str):
                if prev != rel:
                    G[u][v]["rel"] = tuple(sorted({prev, rel}))
            else:
                G[u][v]["rel"] = rel
        else:
            G.add_edge(u, v, rel=rel)

    return G


# ---------------------------------------------------------------------------
# GED cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GedCosts:
    node_ins: float = 1.0
    node_del: float = 1.0
    edge_ins: float = 1.0
    edge_del: float = 1.0

    type_mismatch: float = 2.0
    dir_mismatch: float = 1.0
    dir_unknown: float = 0.25

    # If True, ignore direction for certain types (defaults to ignoring for MeLo/NeLo).
    ignore_dir_for_types: Tuple[str, ...] = ("MeLo", "NeLo")


def make_cost_functions(costs: GedCosts):
    """Return NetworkX-compatible cost callbacks."""

    def node_subst_cost(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        t1 = str(a.get("type") or "UNKNOWN")
        t2 = str(b.get("type") or "UNKNOWN")
        if t1 != t2:
            return float(costs.type_mismatch)

        # Same type: optionally ignore direction for types where direction is often missing/noisy.
        if t1 in costs.ignore_dir_for_types:
            return 0.0

        d1 = str(a.get("direction") or "unknown")
        d2 = str(b.get("direction") or "unknown")
        if d1 == d2:
            return 0.0
        if d1 == "unknown" or d2 == "unknown":
            return float(costs.dir_unknown)
        return float(costs.dir_mismatch)

    def node_del_cost(_: Dict[str, Any]) -> float:
        return float(costs.node_del)

    def node_ins_cost(_: Dict[str, Any]) -> float:
        return float(costs.node_ins)

    def edge_subst_cost(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        ra = a.get("rel")
        rb = b.get("rel")
        # Normalise to tuple for comparison
        if isinstance(ra, str):
            ra_t = (ra,)
        elif isinstance(ra, tuple):
            ra_t = ra
        else:
            ra_t = ("UNKNOWN",)

        if isinstance(rb, str):
            rb_t = (rb,)
        elif isinstance(rb, tuple):
            rb_t = rb
        else:
            rb_t = ("UNKNOWN",)

        return 0.0 if tuple(sorted(ra_t)) == tuple(sorted(rb_t)) else 1.0

    def edge_del_cost(_: Dict[str, Any]) -> float:
        return float(costs.edge_del)

    def edge_ins_cost(_: Dict[str, Any]) -> float:
        return float(costs.edge_ins)

    return node_subst_cost, node_del_cost, node_ins_cost, edge_subst_cost, edge_del_cost, edge_ins_cost


def ged_similarity(
    G1: nx.Graph,
    G2: nx.Graph,
    *,
    costs: GedCosts,
    timeout_s: Optional[float] = None,
) -> Tuple[float, float, float, bool]:
    """Return (score, ged, ged_norm, timed_out).

    score in [0,1] where 1 is identical.
    """
    node_subst, node_del, node_ins, edge_subst, edge_del, edge_ins = make_cost_functions(costs)

    denom = float(G1.number_of_nodes() + G2.number_of_nodes() + G1.number_of_edges() + G2.number_of_edges())
    if denom <= 0:
        return 1.0, 0.0, 0.0, False

    ged = graph_edit_distance(
        G1,
        G2,
        node_subst_cost=node_subst,
        node_del_cost=node_del,
        node_ins_cost=node_ins,
        edge_subst_cost=edge_subst,
        edge_del_cost=edge_del,
        edge_ins_cost=edge_ins,
        timeout=timeout_s,
    )

    if ged is None:
        # Timed out (or failed to converge). Treat as maximally dissimilar.
        return 0.0, float(denom), 1.0, True

    ged_f = float(ged)
    ged_norm = max(0.0, min(1.0, ged_f / denom))
    score = max(0.0, min(1.0, 1.0 - ged_norm))
    return score, ged_f, ged_norm, False


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


def _graph_id(g: JsonGraph) -> str:
    return str(g.get("graph_id") or "")


def _graph_label(g: JsonGraph) -> str:
    return str(g.get("label") or g.get("graph_id") or "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph similarity baseline via exact Graph Edit Distance (GED)")

    p.add_argument("--ist_path", type=str, default=str("data/ist_graphs_all.jsonl"))
    p.add_argument("--templates_path", type=str, default=str("data/lbs_soll_graphs.jsonl"))
    p.add_argument("--out_path", type=str, default=str("data/ged_baseline_matches.jsonl"))

    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_ist", type=int, default=None)
    p.add_argument("--max_templates", type=int, default=None)

    p.add_argument(
        "--directed",
        action="store_true",
        help="Treat edges as directed (default: undirected, consistent with your GNN pipeline).",
    )

    p.add_argument(
        "--timeout_s",
        type=float,
        default=None,
        help="Optional timeout per (Ist,Template) GED computation in seconds. "
        "If set, GED may become inexact but runtime becomes robust.",
    )

    # Optionality directory to enrich template directions from _lbs_optionality
    p.add_argument(
        "--lbs_json_dir",
        type=str,
        default=str("data/lbs_templates"),
        help="Directory containing raw LBS JSONs with _lbs_optionality (used to enrich template directions).",
    )

    # Default ON: evaluate on labelled subset if file exists.
    p.add_argument(
        "--bndl2mc_path",
        type=str,
        default=str("data/training_data/BNDL2MC.csv"),
        help="Optional BNDL2MC.csv for evaluation (default: data/training_data/BNDL2MC.csv). Use '' to disable.",
    )

    # Cost model knobs (kept minimal and interpretable)
    p.add_argument("--type_mismatch_cost", type=float, default=2.0)
    p.add_argument("--dir_mismatch_cost", type=float, default=1.0)
    p.add_argument("--dir_unknown_cost", type=float, default=0.25)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ist_path = _resolve_default(Path(args.ist_path))
    tpl_path = _resolve_default(Path(args.templates_path))
    out_path = Path(args.out_path)
    _ensure_dir(out_path.parent)

    if not ist_path.exists():
        raise FileNotFoundError(f"Ist graphs JSONL not found: {ist_path}")
    if not tpl_path.exists():
        raise FileNotFoundError(f"Template graphs JSONL not found: {tpl_path}")

    ist_graphs = _load_jsonl(ist_path, max_lines=args.max_ist)
    templates = _load_jsonl(tpl_path, max_lines=args.max_templates)
    if not templates:
        raise RuntimeError("No templates loaded.")

    # Load optionality catalog (for enriching template directions).
    lbs_json_dir = _resolve_default(Path(args.lbs_json_dir))
    opt_catalog: Dict[str, Dict[str, Any]] = {}
    if lbs_json_dir.exists() and lbs_json_dir.is_dir():
        opt_catalog = load_lbs_optionality_catalog(lbs_json_dir)
    else:
        print(f"[ged][warn] lbs_json_dir not found / not a directory: {lbs_json_dir} (template dir enrichment disabled)")

    # Optional evaluation labels
    pair_to_mcid: Dict[Tuple[str, str], str] = {}
    bndl_arg = str(args.bndl2mc_path or "").strip()
    if bndl_arg:
        bndl_path = _resolve_default(Path(bndl_arg))
        if bndl_path.exists():
            pair_to_mcid = load_bndl2mc_labels(bndl_path)
            print(f"[ged] Loaded BNDL2MC pairs: {len(pair_to_mcid)} from {bndl_path}")
        else:
            print(f"[ged][warn] BNDL2MC.csv not found at: {bndl_path} (evaluation disabled)")

    print(
        f"[ged] ist_graphs={len(ist_graphs)} | templates={len(templates)} | top_k={args.top_k} | "
        f"directed={bool(args.directed)} | timeout_s={args.timeout_s}"
    )

    costs = GedCosts(
        type_mismatch=float(args.type_mismatch_cost),
        dir_mismatch=float(args.dir_mismatch_cost),
        dir_unknown=float(args.dir_unknown_cost),
    )

    # Pre-build template NX graphs (with direction enrichment from optionality)
    tpl_nx: List[nx.Graph] = []
    tpl_meta: List[Dict[str, Any]] = []
    for t in templates:
        label = _graph_label(t)
        enrich_dirs: Optional[Dict[str, str]] = None

        entry = opt_catalog.get(label)
        if isinstance(entry, dict):
            opt = entry.get("opt")
            if isinstance(opt, dict):
                lbs_objs = opt.get("lbs_objects")
                if isinstance(lbs_objs, list):
                    # object_code -> direction
                    tmp: Dict[str, str] = {}
                    for o in lbs_objs:
                        if not isinstance(o, dict):
                            continue
                        oc = o.get("object_code")
                        if oc is None:
                            continue
                        d = _lbs_object_direction(o)
                        if d and d != "unknown":
                            tmp[str(oc)] = str(d)
                    enrich_dirs = tmp if tmp else None

        Gt = json_to_nx_graph(t, directed=bool(args.directed), enrich_template_dirs=enrich_dirs)
        tpl_nx.append(Gt)
        tpl_meta.append(
            {
                "template_graph_id": _graph_id(t),
                "template_label": label,
                "n_nodes": int(Gt.number_of_nodes()),
                "n_edges": int(Gt.number_of_edges()),
            }
        )

    # Evaluation counters
    eval_total = 0
    eval_top1 = 0
    eval_top3 = 0

    top1_dist: Counter = Counter()
    timed_out_pairs = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for g in ist_graphs:
            Gi = json_to_nx_graph(g, directed=bool(args.directed), enrich_template_dirs=None)
            ist_id = _graph_id(g)

            scored: List[Tuple[float, float, int, float, bool]] = []
            # tuple: (score_desc, ged, idx, ged_norm, timed_out)
            for i, Gt in enumerate(tpl_nx):
                score, ged, ged_norm, to = ged_similarity(Gi, Gt, costs=costs, timeout_s=args.timeout_s)
                if to:
                    timed_out_pairs += 1
                scored.append((score, ged, i, ged_norm, to))

            scored.sort(key=lambda x: (-x[0], x[1], x[2]))
            top_k = max(1, int(args.top_k))
            top = scored[:top_k]

            top_templates: List[Dict[str, Any]] = []
            for rank, (score, ged, idx, ged_norm, to) in enumerate(top, start=1):
                meta = tpl_meta[idx]
                top_templates.append(
                    {
                        "rank": rank,
                        "template_graph_id": meta["template_graph_id"],
                        "template_label": meta["template_label"],
                        "score": float(score),
                        "ged": float(ged),
                        "ged_norm": float(ged_norm),
                        "timed_out": bool(to),
                        "sizes": {
                            "ist_nodes": int(Gi.number_of_nodes()),
                            "ist_edges": int(Gi.number_of_edges()),
                            "tpl_nodes": int(meta["n_nodes"]),
                            "tpl_edges": int(meta["n_edges"]),
                        },
                        "cost_model": {
                            "type_mismatch": float(costs.type_mismatch),
                            "dir_mismatch": float(costs.dir_mismatch),
                            "dir_unknown": float(costs.dir_unknown),
                            "ignore_dir_for_types": list(costs.ignore_dir_for_types),
                        },
                    }
                )

            if top_templates:
                top1_dist[top_templates[0]["template_label"]] += 1

            gt = infer_ground_truth_for_ist(g, pair_to_mcid) if pair_to_mcid else None
            if gt and gt.get("template_label"):
                eval_total += 1
                gt_label = gt["template_label"]
                pred1 = top_templates[0]["template_label"] if top_templates else ""
                if pred1 == gt_label:
                    eval_top1 += 1
                pred_labels = [x["template_label"] for x in top_templates[:3]]
                if gt_label in pred_labels:
                    eval_top3 += 1

            out_obj = {
                "ist_graph_id": ist_id,
                "top_templates": top_templates,
                "ground_truth": gt,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[ged] wrote: {out_path}")

    # Prediction distribution (useful sanity)
    if top1_dist:
        total = sum(top1_dist.values())
        print("[ged] top1 prediction distribution (all Ist graphs):")
        for lbl, cnt in top1_dist.most_common():
            pct = 100.0 * float(cnt) / float(total)
            print(f"  - {lbl}: {cnt} ({pct:.3f}%)")

    if timed_out_pairs > 0:
        print(f"[ged][warn] timed_out_pairs={timed_out_pairs} (consider increasing --timeout_s or leaving it unset for exact GED)")

    if eval_total > 0:
        print(
            "[ged] evaluation on labelled subset | "
            f"n={eval_total} | top1={eval_top1/eval_total:.3f} | top3={eval_top3/eval_total:.3f}"
        )


if __name__ == "__main__":
    main()
