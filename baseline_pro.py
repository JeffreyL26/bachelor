"""baseline_matcher.py

Baseline for matching IST graphs (constructs) to SOLL templates (LBS concepts).

Design goals (Weeks 5–6 in the proposal):
  (a) Rule/Checklist baseline (domain constraints, incl. Min/Max + starr/flexibel)
  (b) Graph-similarity baseline producing a Top-k ranking
  (c) Initial descriptive overview of the dataset instances

Important project constraint:
  - Whenever Min/Max (cardinalities) are needed, this baseline uses the
    `_lbs_optionality` augmentation block (NOT SAP FREQUENCY fields).

This file is intentionally deterministic, explainable, and does not require
training data. It is meant as a strong baseline and a sanity check before
trying DGMC.

Usage (recommended):
  1) Generate IST graphs first (graph_converter.py -> ist_graphs.jsonl)
  2) Place the SOLL template JSONs (9992 ....json) in the project directory
  3) Run:
       python baseline_matcher.py

Outputs:
  - data/ist_baseline_matches.jsonl (Top-k matches per IST graph)
  - data/baseline_descriptive_report.json (dataset + baseline diagnostics)
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# We reuse the existing converter from SAP-export JSON -> template graph.
# NOTE: This conversion ignores `_lbs_optionality` on purpose; we attach
#       `_lbs_optionality` as constraints separately and ONLY use it for Min/Max.
try:
    from graph_templates import lbsjson_to_template_graph
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import graph_templates.lbsjson_to_template_graph. "
        "Ensure baseline_matcher.py is placed next to the project files."
    ) from e


JsonGraph = Dict[str, Any]


# ---------------------------------------------------------------------------
# Step 1: Template constraints from `_lbs_optionality`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OccurrenceConstraint:
    """Aggregated Min/Max constraints for one object_type (MaLo/MeLo/TR/NeLo)."""

    object_type: str
    min_total: int
    max_total: Optional[int]  # None == unbounded ("N")
    flexibilities: Tuple[str, ...] = ()
    raw_entries: Tuple[Dict[str, Any], ...] = ()

    @property
    def is_unbounded(self) -> bool:
        return self.max_total is None


@dataclass
class Template:
    template_id: str
    lbs_code: str
    source_path: str
    graph: JsonGraph
    constraints_by_type: Dict[str, OccurrenceConstraint] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def pattern(self) -> str:
        return str(self.graph.get("graph_attrs", {}).get("pattern", "unknown"))


def _to_int_or_none_max(v: Any) -> Optional[int]:
    """Convert max_occurs: int-like -> int, 'N'/None -> None."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().upper()
        if s == "N":
            return None
        if s.isdigit():
            return int(s)
        # fall back: unknown string
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return None


def parse_optionality_constraints(
    template_json: Dict[str, Any],
) -> Tuple[Dict[str, OccurrenceConstraint], List[str]]:
    """Parse `_lbs_optionality.lbs_objects` into aggregated constraints.

    Returns:
        (constraints_by_type, warnings)
    """

    warnings: List[str] = []
    opt = template_json.get("_lbs_optionality")
    if not isinstance(opt, dict):
        return {}, ["Missing _lbs_optionality block; constraints unavailable."]

    lbs_objects = opt.get("lbs_objects")
    if not isinstance(lbs_objects, list):
        return {}, ["_lbs_optionality.lbs_objects missing or not a list; constraints unavailable."]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in lbs_objects:
        if not isinstance(entry, dict):
            continue
        otype = str(entry.get("object_type") or "").strip()
        if not otype:
            warnings.append("Encountered lbs_object without object_type; ignored.")
            continue
        grouped.setdefault(otype, []).append(entry)

    constraints: Dict[str, OccurrenceConstraint] = {}
    for otype, entries in grouped.items():
        min_total = 0
        max_total: Optional[int] = 0
        flexes: List[str] = []

        for e in entries:
            # Min
            try:
                min_total += int(e.get("min_occurs", 0))
            except Exception:
                warnings.append(f"Non-integer min_occurs for {otype}; treating as 0.")

            # Max
            mx = _to_int_or_none_max(e.get("max_occurs"))
            if mx is None:
                max_total = None
            else:
                if max_total is not None:
                    max_total += mx

            # Flexibility
            flex = str(e.get("flexibility") or "").strip().lower()
            if flex:
                flexes.append(flex)

        constraints[otype] = OccurrenceConstraint(
            object_type=otype,
            min_total=min_total,
            max_total=max_total,
            flexibilities=tuple(sorted(set(flexes))),
            raw_entries=tuple(entries),
        )

    return constraints, warnings


# ---------------------------------------------------------------------------
# Step 2: Graph utilities (signatures, adjacency, node profiles)
# ---------------------------------------------------------------------------


def node_type_counts(g: JsonGraph) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for n in g.get("nodes", []):
        t = str(n.get("type") or "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def edge_type_counts(g: JsonGraph) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in g.get("edges", []):
        r = str(e.get("rel") or "unknown")
        counts[r] = counts.get(r, 0) + 1
    return counts


def undirected_edge_set(g: JsonGraph) -> set[Tuple[str, str, str]]:
    """Undirected edge set (u, v, rel) with u <= v lexicographically."""
    s: set[Tuple[str, str, str]] = set()
    for e in g.get("edges", []):
        u = str(e.get("src"))
        v = str(e.get("dst"))
        rel = str(e.get("rel") or "unknown")
        if not u or not v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        s.add((a, b, rel))
    return s


def adjacency_by_rel(g: JsonGraph) -> Dict[str, List[Tuple[str, str]]]:
    """Adjacency list: node_id -> list of (neighbor_id, rel)."""
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for e in g.get("edges", []):
        u = str(e.get("src"))
        v = str(e.get("dst"))
        rel = str(e.get("rel") or "unknown")
        if not u or not v:
            continue
        adj.setdefault(u, []).append((v, rel))
        adj.setdefault(v, []).append((u, rel))  # treat as undirected for baseline
    return adj


@dataclass(frozen=True)
class NodeProfile:
    node_id: str
    node_type: str
    attrs: Dict[str, Any]
    deg_by_rel: Dict[str, int]


def build_node_profiles(g: JsonGraph) -> Dict[str, NodeProfile]:
    adj = adjacency_by_rel(g)
    profiles: Dict[str, NodeProfile] = {}
    for n in g.get("nodes", []):
        nid = str(n.get("id"))
        if not nid:
            continue
        ntype = str(n.get("type") or "unknown")
        attrs = dict(n.get("attrs") or {})

        deg: Dict[str, int] = {}
        for nb, rel in adj.get(nid, []):
            deg[rel] = deg.get(rel, 0) + 1

        profiles[nid] = NodeProfile(node_id=nid, node_type=ntype, attrs=attrs, deg_by_rel=deg)
    return profiles


@dataclass(frozen=True)
class GraphSignature:
    node_counts: Dict[str, int]
    edge_counts: Dict[str, int]
    tr_per_melo_stats: Tuple[float, float, float]  # (min, mean, max)


def compute_tr_per_melo_stats(g: JsonGraph) -> Tuple[float, float, float]:
    """Compute TR-per-MeLo statistics in the IST graph (or any graph).

    Uses METR edges; counts TR neighbors per MeLo node.
    """

    profiles = build_node_profiles(g)
    melo_ids = [nid for nid, p in profiles.items() if p.node_type == "MeLo"]
    if not melo_ids:
        return (0.0, 0.0, 0.0)
    vals: List[int] = []
    for mid in melo_ids:
        deg_metr = profiles[mid].deg_by_rel.get("METR", 0)
        vals.append(int(deg_metr))
    if not vals:
        return (0.0, 0.0, 0.0)
    return (float(min(vals)), float(sum(vals)) / float(len(vals)), float(max(vals)))


def compute_signature(g: JsonGraph) -> GraphSignature:
    return GraphSignature(
        node_counts=node_type_counts(g),
        edge_counts=edge_type_counts(g),
        tr_per_melo_stats=compute_tr_per_melo_stats(g),
    )


# ---------------------------------------------------------------------------
# Step 3: Candidate filtering
# ---------------------------------------------------------------------------


def range_distance(x: int, min_v: int, max_v: Optional[int]) -> int:
    """Distance of x to a [min_v, max_v] range (0 if inside)."""
    if x < min_v:
        return min_v - x
    if max_v is not None and x > max_v:
        return x - max_v
    return 0


def template_expected_range(
    t: Template, object_type: str, fallback_count: int
) -> Tuple[int, Optional[int]]:
    """Expected (min,max) for one object_type from _lbs_optionality.

    If missing, fall back to the template graph's observed count.
    """
    c = t.constraints_by_type.get(object_type)
    if c is None:
        # No constraints: treat observed count as exact (min=max=count) as weak fallback.
        return fallback_count, fallback_count
    return c.min_total, c.max_total


def template_pattern_from_constraints(t: Template) -> str:
    """Compute a MaLo:MeLo pattern from _lbs_optionality mins if possible."""
    counts = node_type_counts(t.graph)
    malo_min, _ = template_expected_range(t, "MaLo", counts.get("MaLo", 0))
    melo_min, _ = template_expected_range(t, "MeLo", counts.get("MeLo", 0))
    # same convention as graph_converter.classify_pattern
    if malo_min == 1 and melo_min == 1:
        return "1:1"
    if malo_min == 1 and melo_min == 2:
        return "1:2"
    if malo_min == 2 and melo_min == 1:
        return "2:1"
    if malo_min == 2 and melo_min == 2:
        return "2:2"
    return f"{malo_min}:{melo_min}"


def candidate_filter_score(
    ist_sig: GraphSignature,
    templ: Template,
    prefer_same_pattern: bool = True,
) -> float:
    """Lower is better. Used only for candidate pool selection."""

    # Node counts
    t_counts = node_type_counts(templ.graph)

    # We weight MaLo/MeLo higher; TR and NeLo are often flexible.
    weights = {"MaLo": 3.0, "MeLo": 3.0, "TR": 1.0, "NeLo": 1.0}
    dist = 0.0
    for typ, w in weights.items():
        x = int(ist_sig.node_counts.get(typ, 0))
        mn, mx = template_expected_range(templ, typ, int(t_counts.get(typ, 0)))
        dist += w * float(range_distance(x, mn, mx))

    # Edge counts: mild influence
    for rel in ("MEMA", "METR", "MEME", "MENE"):
        dist += 0.2 * abs(int(ist_sig.edge_counts.get(rel, 0)) - int(edge_type_counts(templ.graph).get(rel, 0)))

    # Pattern preference: add penalty if mismatch
    if prefer_same_pattern:
        ist_pat = str(getattr(ist_sig, "pattern", ""))  # not used; pattern is usually in g.graph_attrs
        templ_pat = template_pattern_from_constraints(templ)
        # We cannot access IST pattern here; mismatch penalty handled outside.
        # Still, keep a tiny bias towards templates whose pattern is not "unknown".
        if templ_pat == "unknown":
            dist += 0.5

    return dist


def select_candidate_templates(
    ist_graph: JsonGraph,
    templates: List[Template],
    candidate_pool_size: int = 20,
    allow_pattern_fallback: bool = True,
) -> List[Template]:
    """Select a candidate pool for one IST graph.

    Strategy:
      1) Prefer same MaLo:MeLo pattern (as metadata bucket)
      2) If not enough candidates, fall back to all templates
      3) Within the pool, rank by a cheap filter score
    """

    ist_pat = str(ist_graph.get("graph_attrs", {}).get("pattern", "unknown"))
    ist_sig = compute_signature(ist_graph)

    # Pattern bucket
    in_bucket = [t for t in templates if template_pattern_from_constraints(t) == ist_pat]
    pool = in_bucket
    if allow_pattern_fallback and len(pool) < max(3, candidate_pool_size // 2):
        pool = templates

    scored = sorted(
        ((candidate_filter_score(ist_sig, t), t) for t in pool),
        key=lambda x: x[0],
    )
    return [t for _, t in scored[:candidate_pool_size]]


# ---------------------------------------------------------------------------
# Step 4: Checklist baseline (violations + constraint score)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    object_type: str
    kind: str  # "missing" | "excess" | "edge_missing" | ...
    severity: str  # "hard" | "soft"
    ist_value: Any
    expected_min: int
    expected_max: Optional[int]
    message: str


def _flex_severity(flexibilities: Iterable[str]) -> str:
    flex = {f.lower().strip() for f in flexibilities if f}
    # If any entry is "starr", treat deviations as hard.
    return "hard" if "starr" in flex else "soft"


def checklist_violations(
    ist_graph: JsonGraph,
    templ: Template,
) -> List[Violation]:
    ist_counts = node_type_counts(ist_graph)
    tpl_counts = node_type_counts(templ.graph)

    violations: List[Violation] = []

    # --- A) Occurrence constraints (Min/Max) from _lbs_optionality
    for otype, c in templ.constraints_by_type.items():
        x = int(ist_counts.get(otype, 0))
        mn = int(c.min_total)
        mx = c.max_total

        if x < mn:
            violations.append(
                Violation(
                    object_type=otype,
                    kind="missing",
                    severity="hard" if mn > 0 else "soft",
                    ist_value=x,
                    expected_min=mn,
                    expected_max=mx,
                    message=f"Too few {otype}: got {x}, expected at least {mn}.",
                )
            )
        if mx is not None and x > mx:
            sev = _flex_severity(c.flexibilities)
            violations.append(
                Violation(
                    object_type=otype,
                    kind="excess",
                    severity=sev,
                    ist_value=x,
                    expected_min=mn,
                    expected_max=mx,
                    message=f"Too many {otype}: got {x}, expected at most {mx}.",
                )
            )

    # --- B) Edge evidence checks (soft, because IST may not contain all relation types)
    # If a template graph contains an edge type, we consider it evidence that this
    # relation is conceptually relevant; missing in IST is a weak penalty.
    ist_edges = edge_type_counts(ist_graph)
    tpl_edges = edge_type_counts(templ.graph)
    for rel in ("MEMA", "METR", "MEME", "MENE"):
        if tpl_edges.get(rel, 0) > 0 and ist_edges.get(rel, 0) == 0:
            violations.append(
                Violation(
                    object_type="(graph)",
                    kind="edge_missing",
                    severity="soft",
                    ist_value=0,
                    expected_min=1,
                    expected_max=None,
                    message=f"Template contains relation {rel}, but IST has none (possible missing evidence).",
                )
            )

    # --- C) Fallback sanity warnings (only if constraints missing)
    # If _lbs_optionality is missing for a type that occurs in the template graph,
    # we cannot validate its cardinality.
    for typ in ("MaLo", "MeLo", "TR", "NeLo"):
        if typ not in templ.constraints_by_type and tpl_counts.get(typ, 0) > 0:
            violations.append(
                Violation(
                    object_type=typ,
                    kind="no_constraints",
                    severity="soft",
                    ist_value=int(ist_counts.get(typ, 0)),
                    expected_min=int(tpl_counts.get(typ, 0)),
                    expected_max=int(tpl_counts.get(typ, 0)),
                    message=f"No _lbs_optionality constraints for {typ}; using observed template count as weak fallback.",
                )
            )

    return violations


def constraint_score_from_violations(violations: List[Violation]) -> Tuple[float, bool]:
    """Convert violations into a [0,1] score and a hard-failure flag."""
    hard = any(v.severity == "hard" and v.kind in ("missing", "excess") for v in violations)

    # Penalty model: hard violations matter more.
    penalty = 0.0
    for v in violations:
        if v.kind in ("missing", "excess"):
            delta = abs(int(v.ist_value) - int(v.expected_min)) if v.kind == "missing" else abs(int(v.ist_value) - int(v.expected_max or v.ist_value))
            penalty += (3.0 if v.severity == "hard" else 1.0) * float(max(delta, 1))
        elif v.kind == "edge_missing":
            penalty += 0.5
        elif v.kind == "no_constraints":
            penalty += 0.1

    # Smooth mapping to [0,1]
    score = 1.0 / (1.0 + penalty)

    # If hard violations exist, dampen strongly (still allow ranking output)
    if hard:
        score *= 0.25

    return score, hard


# ---------------------------------------------------------------------------
# Step 5: Similarity baseline (typed partial assignment + edge consistency)
# ---------------------------------------------------------------------------


def _attr_similarity(ist: NodeProfile, tpl: NodeProfile) -> float:
    """Attribute similarity in [0,1].

    Conservative: if attributes are missing, return neutral 0.5.
    """

    if ist.node_type != tpl.node_type:
        return 0.0

    # Default neutral
    score = 0.5

    if ist.node_type == "MaLo":
        d1 = (ist.attrs.get("direction") or "").strip()
        d2 = (tpl.attrs.get("direction") or "").strip()
        if d1 and d2:
            return 1.0 if d1 == d2 else 0.0
        return 0.5

    # Currently IST MeLo/TR/NeLo carry few reliable attrs.
    return score


def _struct_similarity(ist: NodeProfile, tpl: NodeProfile) -> float:
    """Structural similarity in [0,1] based on degree-by-relation."""
    if ist.node_type != tpl.node_type:
        return 0.0
    rels = set(ist.deg_by_rel.keys()) | set(tpl.deg_by_rel.keys())
    if not rels:
        return 1.0
    diff_sum = 0
    for r in rels:
        diff_sum += abs(int(ist.deg_by_rel.get(r, 0)) - int(tpl.deg_by_rel.get(r, 0)))
    return 1.0 / (1.0 + float(diff_sum))


def node_similarity(ist: NodeProfile, tpl: NodeProfile) -> float:
    """Combined node similarity in [0,1]."""
    if ist.node_type != tpl.node_type:
        return 0.0

    a = _attr_similarity(ist, tpl)
    s = _struct_similarity(ist, tpl)

    # Weighted combination: keep attribute conservative, structure helps break ties.
    return 0.6 * a + 0.4 * s


def greedy_typed_partial_assignment(
    ist_graph: JsonGraph,
    tpl_graph: JsonGraph,
) -> Tuple[float, Dict[str, str], Dict[str, float]]:
    """Greedy 1:1 matching per node type.

    Returns:
      - node_score in [0,1]
      - mapping ist_node_id -> tpl_node_id
      - local_similarities per mapped ist_node_id
    """

    ist_profiles = build_node_profiles(ist_graph)
    tpl_profiles = build_node_profiles(tpl_graph)

    # Split nodes by type
    ist_by_type: Dict[str, List[str]] = {}
    tpl_by_type: Dict[str, List[str]] = {}
    for nid, p in ist_profiles.items():
        ist_by_type.setdefault(p.node_type, []).append(nid)
    for nid, p in tpl_profiles.items():
        tpl_by_type.setdefault(p.node_type, []).append(nid)

    mapping: Dict[str, str] = {}
    local_sims: Dict[str, float] = {}
    total_sim = 0.0

    for typ in ("MaLo", "MeLo", "TR", "NeLo"):
        src_nodes = ist_by_type.get(typ, [])
        dst_nodes = tpl_by_type.get(typ, [])
        if not src_nodes or not dst_nodes:
            continue

        pairs: List[Tuple[float, str, str]] = []
        for u in src_nodes:
            for v in dst_nodes:
                sim = node_similarity(ist_profiles[u], tpl_profiles[v])
                pairs.append((sim, u, v))

        pairs.sort(key=lambda x: x[0], reverse=True)
        used_u: set[str] = set()
        used_v: set[str] = set()

        for sim, u, v in pairs:
            if sim <= 0.0:
                break
            if u in used_u or v in used_v:
                continue
            used_u.add(u)
            used_v.add(v)
            mapping[u] = v
            local_sims[u] = sim
            total_sim += sim

    denom = float(max(len(ist_profiles), len(tpl_profiles)) or 1)
    node_score = total_sim / denom
    return node_score, mapping, local_sims


def edge_consistency_score(
    ist_graph: JsonGraph,
    tpl_graph: JsonGraph,
    mapping: Dict[str, str],
) -> float:
    """Compute an edge consistency score in [0,1] for mapped nodes."""

    if not mapping:
        return 0.0

    ist_edges = undirected_edge_set(ist_graph)
    tpl_edges = undirected_edge_set(tpl_graph)

    # IST edges whose endpoints are mapped
    mapped_ist_edges: List[Tuple[str, str, str]] = []
    for (u, v, rel) in ist_edges:
        if u in mapping and v in mapping:
            mapped_ist_edges.append((u, v, rel))

    # Template edges whose endpoints are mapped from some IST nodes
    # Build reverse map: tpl_id -> ist_id (first one wins; collisions imply ambiguity)
    rev: Dict[str, str] = {}
    for u, v in mapping.items():
        if v not in rev:
            rev[v] = u

    mapped_tpl_edges: List[Tuple[str, str, str]] = []
    for (a, b, rel) in tpl_edges:
        if a in rev and b in rev:
            mapped_tpl_edges.append((a, b, rel))

    # Precision: how many mapped IST edges are also present in template after mapping
    correct = 0
    for (u, v, rel) in mapped_ist_edges:
        a = mapping[u]
        b = mapping[v]
        x, y = (a, b) if a <= b else (b, a)
        if (x, y, rel) in tpl_edges:
            correct += 1
    precision = correct / float(len(mapped_ist_edges) or 1)

    # Recall: how many mapped template edges exist in IST
    correct2 = 0
    for (a, b, rel) in mapped_tpl_edges:
        u = rev[a]
        v = rev[b]
        x, y = (u, v) if u <= v else (v, u)
        if (x, y, rel) in ist_edges:
            correct2 += 1
    recall = correct2 / float(len(mapped_tpl_edges) or 1)

    return 0.5 * (precision + recall)


# ---------------------------------------------------------------------------
# Step 6: Final scoring + explanations + Top-k ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    lbs_code: str
    template_id: str
    score: float
    breakdown: Dict[str, float]
    passed_checklist: bool
    hard_violation: bool
    violations: List[Dict[str, Any]]
    mapping: List[Dict[str, Any]]


def match_one_template(
    ist_graph: JsonGraph,
    templ: Template,
    weights: Optional[Dict[str, float]] = None,
) -> MatchResult:
    """Compute baseline score and explanation for (IST, one Template)."""

    weights = weights or {"constraint": 0.4, "node": 0.4, "edge": 0.2}

    # Checklist
    vios = checklist_violations(ist_graph, templ)
    c_score, hard = constraint_score_from_violations(vios)

    # Similarity (typed partial assignment + edge consistency)
    node_score, mapping, local_sims = greedy_typed_partial_assignment(ist_graph, templ.graph)
    edge_score = edge_consistency_score(ist_graph, templ.graph, mapping)

    # Final score
    score = (
        weights["constraint"] * c_score
        + weights["node"] * node_score
        + weights["edge"] * edge_score
    )

    # Passed checklist: no hard missing/excess violations
    passed = not any(v.severity == "hard" and v.kind in ("missing", "excess") for v in vios)

    # Export mapping with local sims
    mapping_out: List[Dict[str, Any]] = []
    for u, v in mapping.items():
        mapping_out.append({"ist": u, "soll": v, "sim": float(local_sims.get(u, 0.0))})

    vios_out: List[Dict[str, Any]] = [
        {
            "object_type": v.object_type,
            "kind": v.kind,
            "severity": v.severity,
            "ist_value": v.ist_value,
            "expected_min": v.expected_min,
            "expected_max": v.expected_max,
            "message": v.message,
        }
        for v in vios
    ]

    breakdown = {"constraint": c_score, "node": node_score, "edge": edge_score}

    return MatchResult(
        lbs_code=templ.lbs_code,
        template_id=templ.template_id,
        score=float(score),
        breakdown={k: float(v) for k, v in breakdown.items()},
        passed_checklist=bool(passed),
        hard_violation=bool(hard),
        violations=vios_out,
        mapping=mapping_out,
    )


def match_ist_graph(
    ist_graph: JsonGraph,
    templates: List[Template],
    top_k: int = 5,
    candidate_pool_size: int = 20,
    allow_pattern_fallback: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[MatchResult], Dict[str, Any]]:
    """Match one IST graph to templates; return top_k results and diagnostics."""

    candidates = select_candidate_templates(
        ist_graph,
        templates,
        candidate_pool_size=candidate_pool_size,
        allow_pattern_fallback=allow_pattern_fallback,
    )

    results = [match_one_template(ist_graph, t, weights=weights) for t in candidates]
    results.sort(key=lambda r: r.score, reverse=True)

    diags = {
        "ist_graph_id": ist_graph.get("graph_id"),
        "ist_pattern": ist_graph.get("graph_attrs", {}).get("pattern"),
        "candidate_pool": len(candidates),
        "candidate_templates": [t.template_id for t in candidates],
    }

    return results[:top_k], diags


# ---------------------------------------------------------------------------
# Step 7: Data loading and reporting
# ---------------------------------------------------------------------------


def load_jsonl_graphs(path: str) -> List[JsonGraph]:
    graphs: List[JsonGraph] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            graphs.append(json.loads(line))
    return graphs


def find_default_ist_path(base_dir: str) -> Optional[str]:
    """Try a few standard locations."""
    candidates = [
        os.path.join(base_dir, "ist_graphs.jsonl"),
        os.path.join(base_dir, "data", "ist_graphs.jsonl"),
        os.path.join(base_dir, "data", "training_data", "ist_graphs.jsonl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_templates_from_directory(base_dir: str) -> List[Template]:
    base_dir = Path(base_dir)
    candidates = [base_dir]
    candidates.append(base_dir / "data" / "lbs_templates")

    template = []
    for c in candidates:
        if c.is_dir():
            # rglob ist robuster, falls Unterordner existieren
            template.extend(sorted(c.rglob("9992*.json")))

    template = list(dict.fromkeys(template))

    if not template:
        raise FileNotFoundError(
            "No template JSONs found. Expected under '.\\data\\lbs_templates' "
            "or the given base directory."
        )

    templates: List[Template] = []
    for p in template:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)

        # LBS code: prefer augmentation, fall back to SAP export
        lbs_code = None
        if isinstance(j.get("_lbs_optionality"), dict):
            lbs_code = j.get("_lbs_optionality", {}).get("lbs_code")
        if not lbs_code:
            try:
                lbs_code = str(j.get("T_MC", [{}])[0].get("LBS_CODE"))
            except Exception:
                lbs_code = None
        lbs_code = str(lbs_code or "unknown")

        # Build template graph from SAP export core structure
        graph = lbsjson_to_template_graph(j)

        # Parse constraints from _lbs_optionality
        constraints, warns = parse_optionality_constraints(j)

        template_id = os.path.basename(p)
        templates.append(
            Template(
                template_id=template_id,
                lbs_code=lbs_code,
                source_path=p,
                graph=graph,
                constraints_by_type=constraints,
                warnings=warns,
            )
        )

    return templates


def build_descriptive_report(
    ist_graphs: List[JsonGraph],
    templates: List[Template],
    match_outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a compact descriptive report for Goal 2."""

    # IST stats
    pat_counts: Dict[str, int] = {}
    n_nodes: List[int] = []
    n_edges: List[int] = []
    rel_counts_total: Dict[str, int] = {}
    tr_stats: List[Tuple[float, float, float]] = []

    for g in ist_graphs:
        pat = str(g.get("graph_attrs", {}).get("pattern", "unknown"))
        pat_counts[pat] = pat_counts.get(pat, 0) + 1
        n_nodes.append(len(g.get("nodes", [])))
        n_edges.append(len(g.get("edges", [])))
        ec = edge_type_counts(g)
        for k, v in ec.items():
            rel_counts_total[k] = rel_counts_total.get(k, 0) + int(v)
        tr_stats.append(compute_tr_per_melo_stats(g))

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / float(len(xs) or 1))

    # Template pattern distribution (constraints-based)
    tpl_pat_counts: Dict[str, int] = {}
    for t in templates:
        p = template_pattern_from_constraints(t)
        tpl_pat_counts[p] = tpl_pat_counts.get(p, 0) + 1

    # Baseline diagnostics: how often hard violations appear among top-1
    hard_top1 = 0
    for row in match_outputs:
        best = row.get("top_k", [None])[0]
        if isinstance(best, dict) and best.get("hard_violation"):
            hard_top1 += 1

    report = {
        "ist": {
            "count": len(ist_graphs),
            "pattern_distribution": pat_counts,
            "nodes": {
                "min": min(n_nodes) if n_nodes else 0,
                "mean": _mean([float(x) for x in n_nodes]) if n_nodes else 0.0,
                "max": max(n_nodes) if n_nodes else 0,
            },
            "edges": {
                "min": min(n_edges) if n_edges else 0,
                "mean": _mean([float(x) for x in n_edges]) if n_edges else 0.0,
                "max": max(n_edges) if n_edges else 0,
            },
            "edge_type_totals": rel_counts_total,
            "tr_per_melo": {
                "min_mean": _mean([t[0] for t in tr_stats]) if tr_stats else 0.0,
                "mean_mean": _mean([t[1] for t in tr_stats]) if tr_stats else 0.0,
                "max_mean": _mean([t[2] for t in tr_stats]) if tr_stats else 0.0,
            },
        },
        "templates": {
            "count": len(templates),
            "pattern_distribution_constraints_based": tpl_pat_counts,
            "templates_with_missing_optionality": sum(1 for t in templates if not t.constraints_by_type),
        },
        "baseline": {
            "top1_hard_violation_count": hard_top1,
            "top1_hard_violation_rate": float(hard_top1) / float(len(match_outputs) or 1),
        },
    }
    return report


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    # Load templates (SOLL)
    templates = load_templates_from_directory(base)
    print(f"Loaded {len(templates)} templates from {base}.")

    # Load IST graphs
    ist_path = find_default_ist_path(base)
    if not ist_path:
        raise FileNotFoundError(
            "IST graphs not found. Generate them first (graph_converter.py) and place 'ist_graphs.jsonl' "
            "in the project directory or in ./data/."
        )
    ist_graphs = load_jsonl_graphs(ist_path)
    print(f"Loaded {len(ist_graphs)} IST graphs from {ist_path}.")

    out_dir = os.path.join(base, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ist_baseline_matches.jsonl")
    report_path = os.path.join(out_dir, "baseline_descriptive_report.json")

    # Baseline parameters
    top_k = 5
    candidate_pool_size = 20

    match_outputs: List[Dict[str, Any]] = []

    with open(out_path, "w", encoding="utf-8") as f_out:
        for g in ist_graphs:
            top, diags = match_ist_graph(
                g,
                templates,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                allow_pattern_fallback=True,
            )

            row = {
                "ist_graph_id": g.get("graph_id"),
                "ist_pattern": g.get("graph_attrs", {}).get("pattern"),
                "top_k": [
                    {
                        "lbs_code": r.lbs_code,
                        "template_id": r.template_id,
                        "score": r.score,
                        "breakdown": r.breakdown,
                        "passed_checklist": r.passed_checklist,
                        "hard_violation": r.hard_violation,
                        "violations": r.violations,
                        "mapping": r.mapping,
                    }
                    for r in top
                ],
                "diagnostics": diags,
            }
            match_outputs.append(row)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_descriptive_report(ist_graphs, templates, match_outputs)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote baseline matches to: {out_path}")
    print(f"Wrote descriptive report to: {report_path}")


if __name__ == "__main__":
    main()
