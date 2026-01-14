"""lbs_rule_baseline.py

Rule / checklist baseline for matching real *Ist*-graphs against LBS template graphs.

Motivation
----------
DGMC is a learned graph matching model. For your thesis you also need a
transparent baseline that:

  * can run directly on `ist_graphs_all.jsonl`
  * uses the structural/cardinality information that originates from
    `_lbs_optionality` (min/max occurrences, mandatory vs optional roles)
  * produces *interpretable* results (checklist-style diagnostics)
  * returns the **top-3** matching templates per Ist graph.

This baseline is intentionally conservative: it only uses information that is
actually present in your current extracted Ist graphs (node types + direction,
and edge relation types). If your Ist graphs omit TR/NeLo/METR/MENE etc., the
baseline cannot "invent" those signals.

Inputs
------
* Ist graphs JSONL (default: ./ist_graphs_all.jsonl)
* Template graphs JSONL (default: ./lbs_soll_graphs.jsonl)
* Optional: BNDL2MC.csv (semicolon-separated) to derive *graph-level* labels for
  a subset of Ist graphs and print evaluation metrics.

Output
------
JSONL where each line corresponds to one Ist graph:

  {
    "ist_graph_id": "...",
    "top_templates": [
      {
        "rank": 1,
        "template_graph_id": "...",
        "template_label": "9992000000042",
        "score": 0.873,
        "breakdown": {"counts": ..., "mandatory": ..., "dirs": ..., "edges": ...},
        "checklist": {...}
      },
      ... up to top_k ...
    ],
    "ground_truth": {"mcid": "S_A1_A2", "template_label": "9992000000042"} | null
  }

Run
---
  python lbs_rule_baseline.py \
      --ist_path ist_graphs_all.jsonl \
      --templates_path lbs_soll_graphs.jsonl \
      --bndl2mc_path BNDL2MC.csv \
      --out_path runs/rule_baseline_matches.jsonl \
      --top_k 3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonGraph = Dict[str, Any]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent


def _resolve_default(path: Path) -> Path:
    """Try a couple of sensible fallbacks."""
    if path.exists():
        return path
    # If user passes "data/foo.jsonl" but repo has "./foo.jsonl"
    alt = BASE_DIR / path.name
    if alt.exists():
        return alt
    return path


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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Direction normalisation (copy of your graph_pipeline logic)
# ---------------------------------------------------------------------------


def _normalize_direction(raw: Any) -> Optional[str]:
    """Normalise direction strings to {consumption, generation, both} or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s_low = s.lower()

    s_up = s.upper()
    if s_up == "Z17":
        return "consumption"
    if s_up == "Z50":
        return "generation"
    if s_up == "Z56":
        return "both"

    if "storage" in s_low or "speicher" in s_low:
        return "both"

    if s_low in ("consumption", "generation", "both"):
        return s_low

    if ("consumption" in s_low and "generation" in s_low) or (
        "einspeis" in s_low and "ausspeis" in s_low
    ):
        return "both"

    if "einspeis" in s_low or "erzeug" in s_low:
        return "generation"
    if "ausspeis" in s_low or "bezug" in s_low or "verbrauch" in s_low:
        return "consumption"

    return None


def _node_direction(node: Dict[str, Any]) -> str:
    """Extract direction in the *same key order* as graph_pipeline."""
    ntype = node.get("type")
    attrs = node.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    raw_dir = attrs.get("direction")
    if raw_dir is None and ntype == "TR":
        raw_dir = attrs.get("tr_direction")
    if raw_dir is None and ntype == "TR":
        raw_dir = attrs.get("tr_type_code") or attrs.get("art_der_technischen_ressource")
    if raw_dir is None and ntype in ("MaLo", "MeLo"):
        raw_dir = attrs.get("direction_hint")

    canon = _normalize_direction(raw_dir)
    return canon if canon is not None else "unknown"


# ---------------------------------------------------------------------------
# Template constraints derived from `_lbs_optionality`
# ---------------------------------------------------------------------------


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_unbounded_max(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().upper() == "N":
        return True
    return False


@dataclass(frozen=True)
class Bounds:
    min: int
    max: Optional[int]  # None => unbounded


@dataclass
class TemplateSignature:
    template_graph_id: str
    template_label: str
    bounds_by_type: Dict[str, Bounds]
    # mandatory roles (min_occurs>=1) split by direction
    mandatory_dir_counts: Dict[str, Counter]
    # overall node counts by type
    node_counts: Counter
    # edge counts by rel
    edge_counts: Counter


@dataclass
class IstSignature:
    ist_graph_id: str
    node_counts: Counter
    dir_counts: Dict[str, Counter]
    edge_counts: Counter
    # attachment coverage for types that appear
    attach_ratio_by_type: Dict[str, float]


def _extract_bounds_from_graph_attrs(ga: Dict[str, Any], prefix: str) -> Bounds:
    mn = _as_int(ga.get(f"{prefix}_min"), 0)
    mx_raw = ga.get(f"{prefix}_max")
    if _is_unbounded_max(mx_raw):
        mx: Optional[int] = None
    else:
        mx = _as_int(mx_raw, None)  # type: ignore[arg-type]
    return Bounds(min=mn, max=mx)


def build_template_signature(tpl: JsonGraph) -> TemplateSignature:
    ga = tpl.get("graph_attrs") or {}
    if not isinstance(ga, dict):
        ga = {}

    template_graph_id = str(tpl.get("graph_id") or "")
    template_label = str(tpl.get("label") or tpl.get("graph_id") or "")

    bounds_by_type: Dict[str, Bounds] = {
        "MaLo": _extract_bounds_from_graph_attrs(ga, "malo"),
        "MeLo": _extract_bounds_from_graph_attrs(ga, "melo"),
        "TR": _extract_bounds_from_graph_attrs(ga, "tr"),
        "NeLo": _extract_bounds_from_graph_attrs(ga, "nelo"),
    }

    node_counts = Counter()
    mandatory_dir_counts: Dict[str, Counter] = defaultdict(Counter)
    for n in tpl.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        t = str(n.get("type"))
        if t not in ("MaLo", "MeLo", "TR", "NeLo"):
            continue
        node_counts[t] += 1

        attrs = n.get("attrs") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        if _as_int(attrs.get("min_occurs"), 0) >= 1:
            d = _node_direction(n)
            mandatory_dir_counts[t][d] += 1

    edge_counts = Counter()
    for e in tpl.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
        if isinstance(rel, str):
            rel = rel.strip().upper()
        edge_counts[str(rel)] += 1

    return TemplateSignature(
        template_graph_id=template_graph_id,
        template_label=template_label,
        bounds_by_type=bounds_by_type,
        mandatory_dir_counts=dict(mandatory_dir_counts),
        node_counts=node_counts,
        edge_counts=edge_counts,
    )


def build_ist_signature(g: JsonGraph) -> IstSignature:
    ist_graph_id = str(g.get("graph_id") or "")

    node_counts = Counter()
    dir_counts: Dict[str, Counter] = defaultdict(Counter)

    malos: List[str] = []
    melos: List[str] = []
    trs: List[str] = []
    nelos: List[str] = []

    for n in g.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        t = str(n.get("type"))
        if t not in ("MaLo", "MeLo", "TR", "NeLo"):
            continue
        node_counts[t] += 1
        d = _node_direction(n)
        dir_counts[t][d] += 1

        nid = n.get("id")
        if nid is None:
            continue
        sid = str(nid)
        if t == "MaLo":
            malos.append(sid)
        elif t == "MeLo":
            melos.append(sid)
        elif t == "TR":
            trs.append(sid)
        elif t == "NeLo":
            nelos.append(sid)

    # edge rel counts (directed JSON edges)
    edge_counts = Counter()
    edges = [e for e in (g.get("edges", []) or []) if isinstance(e, dict)]
    for e in edges:
        rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
        if isinstance(rel, str):
            rel = rel.strip().upper()
        edge_counts[str(rel)] += 1

    # attachment coverage: for each node of a given type, does it have at least one edge
    # of the expected rel to a MeLo?
    melo_set = set(melos)

    # Build adjacency by rel (undirected check)
    def has_rel_to_melo(node_id: str, rel: str) -> bool:
        rel = rel.upper()
        for e in edges:
            r = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
            if isinstance(r, str):
                r = r.strip().upper()
            if str(r) != rel:
                continue
            s = str(e.get("src"))
            d = str(e.get("dst"))
            if s == node_id and d in melo_set:
                return True
            if d == node_id and s in melo_set:
                return True
        return False

    attach_ratio_by_type: Dict[str, float] = {}
    if malos:
        ok = sum(1 for nid in malos if has_rel_to_melo(nid, "MEMA"))
        attach_ratio_by_type["MaLo"] = ok / max(1, len(malos))
    if trs:
        ok = sum(1 for nid in trs if has_rel_to_melo(nid, "METR"))
        attach_ratio_by_type["TR"] = ok / max(1, len(trs))
    if nelos:
        ok = sum(1 for nid in nelos if has_rel_to_melo(nid, "MENE"))
        attach_ratio_by_type["NeLo"] = ok / max(1, len(nelos))

    return IstSignature(
        ist_graph_id=ist_graph_id,
        node_counts=node_counts,
        dir_counts=dict(dir_counts),
        edge_counts=edge_counts,
        attach_ratio_by_type=attach_ratio_by_type,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _bounded_satisfaction(val: int, b: Bounds) -> float:
    """Score in [0,1] for satisfying a min/max range."""
    if val < b.min:
        if b.min <= 0:
            return 1.0
        return max(0.0, min(1.0, val / b.min))
    if b.max is None:
        return 1.0
    if val > b.max:
        if b.max <= 0:
            return 0.0
        return max(0.0, min(1.0, b.max / val))
    return 1.0


def _specificity_factor(b: Bounds) -> float:
    """Return a mild factor in [0.5, 1.0] that prefers tighter bounds.

    Why this matters: two templates may *allow* an observed count, but one may be
    far more specific. Example:

      - Template A: MaLo in [1, 2]
      - Template B: MaLo in [2, 2]

    If the instance has 2 MaLos, both templates are feasible. For a
    rule-based baseline we should (slightly) prefer the tighter template B.

    The factor is intentionally mild (never below 0.5) so we don't destroy
    matching when a template has open-ended bounds (max=None) from "N".
    """

    if b.max is None:
        # Open-ended range -> low specificity, but keep it mild.
        return 0.70

    width = max(0, int(b.max) - int(b.min))
    # width=0 => spec=1.0, width=1 =>0.5, width=2 =>0.333...
    spec = 1.0 / (1.0 + float(width))
    # Map spec in (0,1] to factor in [0.5,1.0]
    return 0.5 + 0.5 * spec


def _dir_overlap_score(req: Counter, obs: Counter) -> float:
    """Fraction of required direction-roles that can be covered by observations."""
    total = sum(req.values())
    if total == 0:
        return 1.0
    matched = 0
    for d, k in req.items():
        matched += min(int(k), int(obs.get(d, 0)))
    return matched / total


@dataclass
class ScoreBreakdown:
    counts: float
    mandatory: float
    dirs: float
    edges: float
    attachments: float
    total: float


def score_pair(
    ist: IstSignature,
    tpl: TemplateSignature,
    *,
    w_counts: float = 0.55,
    w_mandatory: float = 0.25,
    w_dirs: float = 0.15,
    w_edges: float = 0.03,
    w_attach: float = 0.02,
) -> ScoreBreakdown:
    """Compute a transparent similarity score and a breakdown."""

    # 1) Cardinalities / bounds (mostly driven by MaLo/MeLo)
    type_weights = {"MaLo": 0.45, "MeLo": 0.45, "TR": 0.05, "NeLo": 0.05}
    s_counts = 0.0
    w_sum = 0.0
    for t, wt in type_weights.items():
        val = int(ist.node_counts.get(t, 0))
        b = tpl.bounds_by_type.get(t, Bounds(0, None))
        # Within-bounds is necessary, but not always sufficient: prefer templates
        # with tighter (more specific) cardinality intervals when both are feasible.
        s_counts += wt * _bounded_satisfaction(val, b) * _specificity_factor(b)
        w_sum += wt
    s_counts = s_counts / max(1e-9, w_sum)

    # 2) Mandatory roles (min_occurs>=1) coverage (type+direction)
    #    This is where `_lbs_optionality` becomes discriminative.
    mand_type_weights = {"MaLo": 0.70, "MeLo": 0.25, "TR": 0.03, "NeLo": 0.02}
    s_mand = 0.0
    w_sum = 0.0
    for t, wt in mand_type_weights.items():
        req = tpl.mandatory_dir_counts.get(t, Counter())
        obs = ist.dir_counts.get(t, Counter())
        s_mand += wt * _dir_overlap_score(req, obs)
        w_sum += wt
    s_mand = s_mand / max(1e-9, w_sum)

    # 3) Direction distribution similarity (soft, not only mandatory)
    #    Use only MaLo/TR heavily; MeLo directions are often missing in Ist graphs.
    dir_type_weights = {"MaLo": 0.75, "TR": 0.20, "MeLo": 0.05}
    s_dirs = 0.0
    w_sum = 0.0
    for t, wt in dir_type_weights.items():
        # compare template's *total* direction counts (incl. optional) to Ist
        # We approximate template total direction counts by taking mandatory counts
        # plus the remaining nodes of that type as "unknown".
        req = Counter(tpl.mandatory_dir_counts.get(t, Counter()))
        extra = int(tpl.node_counts.get(t, 0) - sum(req.values()))
        if extra > 0:
            req["unknown"] += extra
        obs = ist.dir_counts.get(t, Counter())
        s_dirs += wt * _dir_overlap_score(req, obs)
        w_sum += wt
    s_dirs = s_dirs / max(1e-9, w_sum)

    # 4) Edge sanity (only for *mandatory* endpoint types)
    #    In your current Ist graphs, typically only MEMA exists.
    req_malo = sum(tpl.mandatory_dir_counts.get("MaLo", Counter()).values())
    req_tr = sum(tpl.mandatory_dir_counts.get("TR", Counter()).values())
    req_nelo = sum(tpl.mandatory_dir_counts.get("NeLo", Counter()).values())
    required_edges = {
        "MEMA": req_malo,
        "METR": req_tr,
        "MENE": req_nelo,
    }
    edge_scores = []
    for rel, req_cnt in required_edges.items():
        if req_cnt <= 0:
            continue
        obs_cnt = int(ist.edge_counts.get(rel, 0))
        edge_scores.append(min(obs_cnt, req_cnt) / req_cnt)
    s_edges = float(sum(edge_scores) / max(1, len(edge_scores))) if edge_scores else 1.0

    # 5) Attachment coverage (only if Ist contains the node types)
    attach_scores = []
    for t, ratio in ist.attach_ratio_by_type.items():
        # If template *cannot* have that type at all (max==0), treat as mismatch.
        b = tpl.bounds_by_type.get(t)
        if b is not None and b.max == 0 and ratio > 0:
            attach_scores.append(0.0)
        else:
            attach_scores.append(float(ratio))
    s_attach = float(sum(attach_scores) / max(1, len(attach_scores))) if attach_scores else 1.0

    total = (
        w_counts * s_counts
        + w_mandatory * s_mand
        + w_dirs * s_dirs
        + w_edges * s_edges
        + w_attach * s_attach
    )
    return ScoreBreakdown(
        counts=s_counts,
        mandatory=s_mand,
        dirs=s_dirs,
        edges=s_edges,
        attachments=s_attach,
        total=float(total),
    )


# ---------------------------------------------------------------------------
# Optional: label mapping from BNDL2MC.csv
# ---------------------------------------------------------------------------


MCID_TO_TEMPLATE_LABEL = {
    # Provided mapping (project knowledge)
    "S_A1_A2": "9992000000042",
    "S_C3": "9992000000175",
    "S_A001": "9992000000026",
}


def load_bndl2mc_labels(bndl2mc_path: Path) -> Dict[Tuple[str, str], str]:
    """Return map (malo_id, melo_id) -> MCID."""
    # semicolon-separated, columns: Bündel;Marktlokation;Messlokation;MCID
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
    g: JsonGraph,
    pair_to_mcid: Dict[Tuple[str, str], str],
) -> Optional[Dict[str, str]]:
    """If any (MaLo,MeLo) pair of the graph occurs in BNDL2MC, return ground truth."""
    malos = [str(n.get("id")) for n in (g.get("nodes") or []) if isinstance(n, dict) and n.get("type") == "MaLo" and n.get("id") is not None]
    melos = [str(n.get("id")) for n in (g.get("nodes") or []) if isinstance(n, dict) and n.get("type") == "MeLo" and n.get("id") is not None]
    for malo in malos:
        for melo in melos:
            mcid = pair_to_mcid.get((malo, melo))
            if mcid:
                return {"mcid": mcid, "template_label": MCID_TO_TEMPLATE_LABEL.get(mcid, "")}
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule/checklist baseline for Ist→Template matching")
    p.add_argument("--ist_path", type=str, default=str(BASE_DIR / "ist_graphs_all.jsonl"))
    p.add_argument("--templates_path", type=str, default=str(BASE_DIR / "lbs_soll_graphs.jsonl"))
    p.add_argument("--out_path", type=str, default=str(BASE_DIR / "runs" / "rule_baseline_matches.jsonl"))
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_ist", type=int, default=None)
    p.add_argument("--max_templates", type=int, default=None)
    p.add_argument("--bndl2mc_path", type=str, default=None, help="Optional BNDL2MC.csv for evaluation")
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

    # Optional labels
    pair_to_mcid: Dict[Tuple[str, str], str] = {}
    if args.bndl2mc_path:
        bndl_path = _resolve_default(Path(args.bndl2mc_path))
        if not bndl_path.exists():
            raise FileNotFoundError(f"BNDL2MC.csv not found: {bndl_path}")
        pair_to_mcid = load_bndl2mc_labels(bndl_path)
        print(f"[baseline] Loaded BNDL2MC pairs: {len(pair_to_mcid)}")

    print(f"[baseline] ist_graphs={len(ist_graphs)} | templates={len(templates)} | top_k={args.top_k}")

    tpl_sigs = [build_template_signature(t) for t in templates]

    # Evaluation counters (if labels exist)
    eval_total = 0
    eval_top1 = 0
    eval_top3 = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for g in ist_graphs:
            ist_sig = build_ist_signature(g)

            scored: List[Tuple[float, int, ScoreBreakdown]] = []
            for i, tpl_sig in enumerate(tpl_sigs):
                bd = score_pair(ist_sig, tpl_sig)
                scored.append((bd.total, i, bd))

            scored.sort(key=lambda x: (-x[0], x[1]))
            top_k = max(1, int(args.top_k))
            top = scored[:top_k]

            top_templates = []
            for rank, (score, idx, bd) in enumerate(top, start=1):
                tpl = templates[idx]
                tpl_sig = tpl_sigs[idx]

                # Build checklist (interpretable)
                checklist = {
                    "node_counts": {
                        t: {
                            "obs": int(ist_sig.node_counts.get(t, 0)),
                            "min": int(tpl_sig.bounds_by_type[t].min),
                            "max": (None if tpl_sig.bounds_by_type[t].max is None else int(tpl_sig.bounds_by_type[t].max)),
                            "ok": bool(
                                ist_sig.node_counts.get(t, 0) >= tpl_sig.bounds_by_type[t].min
                                and (
                                    tpl_sig.bounds_by_type[t].max is None
                                    or ist_sig.node_counts.get(t, 0) <= tpl_sig.bounds_by_type[t].max
                                )
                            ),
                        }
                        for t in ("MaLo", "MeLo", "TR", "NeLo")
                    },
                    "mandatory_dirs": {
                        t: {
                            "required": dict(tpl_sig.mandatory_dir_counts.get(t, Counter())),
                            "observed": dict(ist_sig.dir_counts.get(t, Counter())),
                        }
                        for t in ("MaLo", "MeLo", "TR", "NeLo")
                        if sum(tpl_sig.mandatory_dir_counts.get(t, Counter()).values()) > 0
                    },
                    "edge_counts": {
                        "observed": {k: int(v) for k, v in ist_sig.edge_counts.items()},
                        "template": {k: int(v) for k, v in tpl_sig.edge_counts.items()},
                    },
                    "attachment_coverage": {k: float(v) for k, v in ist_sig.attach_ratio_by_type.items()},
                }

                top_templates.append(
                    {
                        "rank": rank,
                        "template_graph_id": tpl.get("graph_id"),
                        "template_label": tpl_sig.template_label,
                        "score": float(score),
                        "breakdown": {
                            "counts": bd.counts,
                            "mandatory": bd.mandatory,
                            "dirs": bd.dirs,
                            "edges": bd.edges,
                            "attachments": bd.attachments,
                        },
                        "checklist": checklist,
                    }
                )

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
                "ist_graph_id": ist_sig.ist_graph_id,
                "top_templates": top_templates,
                "ground_truth": gt,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[baseline] wrote: {out_path}")
    if eval_total > 0:
        print(
            "[baseline] evaluation on labelled subset | "
            f"n={eval_total} | top1={eval_top1/eval_total:.3f} | top3={eval_top3/eval_total:.3f}"
        )


if __name__ == "__main__":
    main()
