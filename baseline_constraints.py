"""
baseline_constraints.py  (aka lbs_rule_baseline.py)

Rule / checklist baseline for matching real *Ist*-graphs against LBS template graphs.

Changes in this revision (requested)
------------------------------------
- `_node_direction` is now **pipeline-conform** to the current `graph_pipeline.py`
  (key order: direction -> TR: tr_direction -> TR: tr_type_code/art_der_technischen_ressource
  -> MaLo/MeLo: direction_hint).
- Template directions are extracted robustly from `_lbs_optionality.lbs_objects`
  (try multiple keys, not only `direction`).
- min/max cardinalities are enforced as a **hard constraint**: templates that violate
  bounds (for any of MaLo/MeLo/TR/NeLo) are ranked behind feasible templates and get
  `score=0.0` (but still appear if *no* template is feasible, so you always receive Top-3).
- `--bndl2mc_path` defaults to `data/BNDL2MC.csv` so you get performance feedback
  immediately (if the file exists; otherwise a warning is printed and evaluation is skipped).

Run
---
  python baseline_constraints.py \
      --ist_path data/ist_graphs_all.jsonl \
      --templates_path data/lbs_soll_graphs.jsonl \
      --lbs_json_dir data/lbs_templates \
      --out_path data/rule_baseline_matches.jsonl \
      --top_k 3
"""

from __future__ import annotations

import argparse
import json
import re
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

    # TR-Art-Codes
    s_up = s.upper()
    if s_up == "Z17":
        return "consumption"
    if s_up == "Z50":
        return "generation"
    if s_up == "Z56":
        return "both"

    # Storage/Speicher
    if "storage" in s_low or "speicher" in s_low:
        return "both"

    # canonical
    if s_low in ("consumption", "generation", "both"):
        return s_low

    # combined
    if ("consumption" in s_low and "generation" in s_low) or (
        "einspeis" in s_low and "ausspeis" in s_low
    ):
        return "both"

    # German heuristics
    if "einspeis" in s_low or "erzeug" in s_low:
        return "generation"
    if "ausspeis" in s_low or "bezug" in s_low or "verbrauch" in s_low:
        return "consumption"

    return None


def _node_direction(node: Dict[str, Any]) -> str:
    """Extract direction in the *same key order* as the current `graph_pipeline.py`."""
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


def _lbs_object_direction(obj: Dict[str, Any]) -> str:
    """Extract direction for LBS optionality objects robustly.

    In practice, different extracts may use different keys. We try a small set
    in a stable order and normalise with `_normalize_direction`.
    """
    if not isinstance(obj, dict):
        return "unknown"

    t = str(obj.get("object_type") or "").strip()

    raw = obj.get("direction")
    if raw is None:
        raw = obj.get("direction_hint")

    # TR-specific fallbacks
    if raw is None and t == "TR":
        raw = obj.get("tr_direction")
    if raw is None and t == "TR":
        raw = obj.get("tr_type_code") or obj.get("art_der_technischen_ressource")

    # Generic fallback: some JSONs store direction-like info under attrs
    if raw is None:
        attrs = obj.get("attrs")
        if isinstance(attrs, dict):
            raw = attrs.get("direction") or attrs.get("direction_hint")

    canon = _normalize_direction(raw)
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


def _extract_version_from_filename(name: str) -> int:
    """Extract "VerX" from filenames like "9992 ... - Ver3.json"."""
    m = re.search(r"\bVer\s*(\d+)\b", name, flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def load_lbs_optionality_catalog(lbs_json_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load mapping lbs_code -> {_lbs_optionality, path, version}.

    The project directory typically contains multiple versions per LBS code.
    We keep the highest version number (VerX) as the most recent.
    """
    catalog: Dict[str, Dict[str, Any]] = {}

    for path in sorted(lbs_json_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        opt = obj.get("_lbs_optionality")
        if not isinstance(opt, dict):
            continue

        lbs_code = str(opt.get("lbs_code") or "").strip()
        if not lbs_code:
            continue

        ver = _extract_version_from_filename(path.name)

        prev = catalog.get(lbs_code)
        if prev is None:
            catalog[lbs_code] = {"opt": opt, "path": str(path), "version": ver}
            continue

        # Prefer higher version. If equal, prefer entries that actually contain
        # a non-empty constraints dict.
        prev_ver = int(prev.get("version") or 0)
        if ver > prev_ver:
            catalog[lbs_code] = {"opt": opt, "path": str(path), "version": ver}
            continue

        if ver == prev_ver:
            prev_cons = (prev.get("opt") or {}).get("constraints")
            cur_cons = opt.get("constraints")
            prev_has = isinstance(prev_cons, dict) and bool(prev_cons)
            cur_has = isinstance(cur_cons, dict) and bool(cur_cons)
            if cur_has and not prev_has:
                catalog[lbs_code] = {"opt": opt, "path": str(path), "version": ver}

    return catalog


@dataclass(frozen=True)
class Bounds:
    min: int
    max: Optional[int]  # None => unbounded


def _bounds_from_lbs_objects(lbs_objects: List[Dict[str, Any]], object_type: str) -> Bounds:
    """Aggregate min/max occurrences for a node type across all LBS roles."""
    mn = 0
    mx_sum = 0
    unbounded = False

    for o in lbs_objects:
        if not isinstance(o, dict):
            continue
        if str(o.get("object_type")) != object_type:
            continue

        mn += _as_int(o.get("min_occurs"), 0)

        mx_raw = o.get("max_occurs")
        if _is_unbounded_max(mx_raw):
            unbounded = True
        else:
            mx_sum += _as_int(mx_raw, 0)

    return Bounds(min=int(mn), max=None if unbounded else int(mx_sum))


def _extract_bounds_from_graph_attrs(ga: Dict[str, Any], prefix: str) -> Bounds:
    mn = _as_int(ga.get(f"{prefix}_min"), 0)
    mx_raw = ga.get(f"{prefix}_max")
    if _is_unbounded_max(mx_raw):
        mx: Optional[int] = None
    else:
        try:
            mx = int(mx_raw)
        except Exception:
            mx = None
    return Bounds(min=mn, max=mx)


def _ref_to_melo_candidates(raw: Any) -> List[str]:
    """Normalise reference_to_melo into a list of candidate MeLo object_codes."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, (list, tuple)):
        out: List[str] = []
        for x in raw:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(raw).strip()
    return [s] if s else []


def _extract_equal_count_constraints(
    constraints: Any,
    code_to_type: Dict[str, str],
) -> List[Dict[str, str]]:
    """Extract equal-count constraints in a uniform format.

    We only *score* them on the **type-level** (MaLo/MeLo/TR/NeLo) because Ist
    graphs do not contain template object_codes.
    """
    out: List[Dict[str, str]] = []

    if not isinstance(constraints, dict):
        return out

    cc = constraints.get("cardinality_constraints")
    if not isinstance(cc, list):
        return out

    for item in cc:
        if not isinstance(item, dict):
            continue
        pair = item.get("equal_count_between_object_codes")
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            continue
        a = str(pair[0])
        b = str(pair[1])
        ta = code_to_type.get(a, "")
        tb = code_to_type.get(b, "")
        if not ta or not tb:
            continue
        out.append({"a_code": a, "a_type": ta, "b_code": b, "b_type": tb})

    return out


@dataclass
class TemplateSignature:
    template_graph_id: str
    template_label: str

    # Type-level bounds (sum of min/max occurrences across all roles)
    bounds_by_type: Dict[str, Bounds]

    # Mandatory roles (min_occurs>=1) split by direction, counted in **min occurrences**
    mandatory_dir_counts: Dict[str, Counter]

    # Number of roles (not occurrences) by type
    role_counts: Counter

    # Edge rel counts from the template graph JSONL (may be 0 if edges omitted)
    edge_counts: Counter

    # Optionality linkage
    optionality_source_path: Optional[str]
    optionality_version: Optional[int]

    # Reference / structure signals from `_lbs_optionality`
    mandatory_melo_roles: List[str]  # MeLo roles with min_occurs>=1 (object_codes)
    melo_min_attach_reqs: Dict[str, Counter]  # melo_code -> Counter(type -> min_required)
    ambiguous_attach_reqs: List[Dict[str, Any]]  # constraints with reference_to_melo lists

    # Extracted constraint summaries (reported; only partially scorable)
    reference_rules: List[Dict[str, Any]]
    equal_count_constraints: List[Dict[str, str]]


@dataclass
class IstSignature:
    ist_graph_id: str
    node_counts: Counter
    dir_counts: Dict[str, Counter]
    edge_counts: Counter

    # attachment coverage for types that appear
    attach_ratio_by_type: Dict[str, float]

    # structure: for each MeLo node, how many unique neighbours of each type
    melo_neighbour_counts: Dict[str, Counter]


def build_template_signature(
    tpl: JsonGraph,
    *,
    optionality_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TemplateSignature:
    """Build a template signature; prefer `_lbs_optionality` from catalog."""

    ga = tpl.get("graph_attrs") or {}
    if not isinstance(ga, dict):
        ga = {}

    template_graph_id = str(tpl.get("graph_id") or "")
    template_label = str(tpl.get("label") or tpl.get("graph_id") or "")

    # --- Optionality lookup (source of truth) ---
    opt: Optional[Dict[str, Any]] = None
    opt_path: Optional[str] = None
    opt_ver: Optional[int] = None
    lbs_objects: List[Dict[str, Any]] = []

    if optionality_catalog is not None:
        entry = optionality_catalog.get(template_label)
        if entry is not None:
            opt = entry.get("opt") if isinstance(entry, dict) else None
            if isinstance(opt, dict):
                lbs_objects_raw = opt.get("lbs_objects")
                if isinstance(lbs_objects_raw, list):
                    lbs_objects = [x for x in lbs_objects_raw if isinstance(x, dict)]
                opt_path = str(entry.get("path") or "") or None
                try:
                    opt_ver = int(entry.get("version"))
                except Exception:
                    opt_ver = None

    # --- Bounds (min/max occurrences) ---
    if lbs_objects:
        bounds_by_type: Dict[str, Bounds] = {
            "MaLo": _bounds_from_lbs_objects(lbs_objects, "MaLo"),
            "MeLo": _bounds_from_lbs_objects(lbs_objects, "MeLo"),
            "TR": _bounds_from_lbs_objects(lbs_objects, "TR"),
            "NeLo": _bounds_from_lbs_objects(lbs_objects, "NeLo"),
        }
    else:
        # Fallback: graph_attrs produced by your graph builder
        bounds_by_type = {
            "MaLo": _extract_bounds_from_graph_attrs(ga, "malo"),
            "MeLo": _extract_bounds_from_graph_attrs(ga, "melo"),
            "TR": _extract_bounds_from_graph_attrs(ga, "tr"),
            "NeLo": _extract_bounds_from_graph_attrs(ga, "nelo"),
        }

    # --- Role counts + mandatory direction counts ---
    role_counts = Counter()
    mandatory_dir_counts: Dict[str, Counter] = defaultdict(Counter)

    if lbs_objects:
        for o in lbs_objects:
            t = str(o.get("object_type") or "")
            if t not in ("MaLo", "MeLo", "TR", "NeLo"):
                continue
            role_counts[t] += 1

            mn = _as_int(o.get("min_occurs"), 0)
            if mn >= 1:
                d = _lbs_object_direction(o)
                mandatory_dir_counts[t][d] += mn
    else:
        # Fallback: from template graph nodes
        for n in tpl.get("nodes", []) or []:
            if not isinstance(n, dict):
                continue
            t = str(n.get("type"))
            if t not in ("MaLo", "MeLo", "TR", "NeLo"):
                continue
            role_counts[t] += 1

            attrs = n.get("attrs") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            if _as_int(attrs.get("min_occurs"), 0) >= 1:
                d = _node_direction(n)
                mandatory_dir_counts[t][d] += 1

    # --- Template edges (for reporting only; may be empty for ambiguous templates) ---
    edge_counts = Counter()
    for e in tpl.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
        if isinstance(rel, str):
            rel = rel.strip().upper()
        edge_counts[str(rel)] += 1

    # --- Reference + structure extraction from optionality ---
    mandatory_melo_roles: List[str] = []
    melo_min_attach_reqs: Dict[str, Counter] = defaultdict(Counter)
    ambiguous_attach_reqs: List[Dict[str, Any]] = []

    reference_rules: List[Dict[str, Any]] = []
    equal_count_constraints: List[Dict[str, str]] = []

    if lbs_objects:
        code_to_type = {str(o.get("object_code")): str(o.get("object_type")) for o in lbs_objects}

        # Mandatory MeLo roles
        for o in lbs_objects:
            if str(o.get("object_type")) != "MeLo":
                continue
            if _as_int(o.get("min_occurs"), 0) >= 1:
                mandatory_melo_roles.append(str(o.get("object_code")))

        # Attachment requirements (only *mandatory* occurrences, and only unambiguous references)
        for o in lbs_objects:
            t = str(o.get("object_type") or "")
            if t not in ("MaLo", "TR", "NeLo"):
                continue

            mn = _as_int(o.get("min_occurs"), 0)
            if mn < 1:
                continue

            candidates = _ref_to_melo_candidates(o.get("reference_to_melo"))
            if not candidates:
                continue

            if len(candidates) == 1:
                melo = candidates[0]
                melo_min_attach_reqs[melo][t] += mn
            else:
                # Ambiguous: the role may attach to one-of-several MeLos.
                ambiguous_attach_reqs.append(
                    {
                        "object_code": str(o.get("object_code")),
                        "object_type": t,
                        "min_occurs": mn,
                        "candidates": candidates,
                    }
                )

        # Constraints block (if present)
        cons = (opt or {}).get("constraints") if isinstance(opt, dict) else None
        if isinstance(cons, dict):
            rr = cons.get("reference_rules")
            if isinstance(rr, list):
                reference_rules = [x for x in rr if isinstance(x, dict)]
            equal_count_constraints = _extract_equal_count_constraints(cons, code_to_type)

    return TemplateSignature(
        template_graph_id=template_graph_id,
        template_label=template_label,
        bounds_by_type=bounds_by_type,
        mandatory_dir_counts=dict(mandatory_dir_counts),
        role_counts=role_counts,
        edge_counts=edge_counts,
        optionality_source_path=opt_path,
        optionality_version=opt_ver,
        mandatory_melo_roles=mandatory_melo_roles,
        melo_min_attach_reqs=dict(melo_min_attach_reqs),
        ambiguous_attach_reqs=ambiguous_attach_reqs,
        reference_rules=reference_rules,
        equal_count_constraints=equal_count_constraints,
    )


def build_ist_signature(g: JsonGraph) -> IstSignature:
    ist_graph_id = str(g.get("graph_id") or "")

    node_counts = Counter()
    dir_counts: Dict[str, Counter] = defaultdict(Counter)

    id_to_type: Dict[str, str] = {}
    malos: List[str] = []
    melos: List[str] = []
    trs: List[str] = []
    nelos: List[str] = []

    nodes = [n for n in (g.get("nodes") or []) if isinstance(n, dict)]
    for n in nodes:
        t = str(n.get("type"))
        nid = n.get("id")
        if nid is None:
            continue
        sid = str(nid)
        id_to_type[sid] = t

        if t not in ("MaLo", "MeLo", "TR", "NeLo"):
            continue

        node_counts[t] += 1
        d = _node_direction(n)
        dir_counts[t][d] += 1

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

    # structure: per MeLo node, count unique neighbours of each type via (MEMA/METR/MENE)
    neighbour_sets: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    rel_to_expected = {"MEMA": "MaLo", "METR": "TR", "MENE": "NeLo"}
    for e in edges:
        rel = e.get("rel") or e.get("type") or e.get("edge_type") or e.get("relation")
        if isinstance(rel, str):
            rel = rel.strip().upper()
        rel = str(rel)
        if rel not in rel_to_expected:
            continue
        exp_type = rel_to_expected[rel]

        s = str(e.get("src"))
        d = str(e.get("dst"))
        st = id_to_type.get(s)
        dt = id_to_type.get(d)

        if st == "MeLo" and dt == exp_type:
            neighbour_sets[s][exp_type].add(d)
        elif dt == "MeLo" and st == exp_type:
            neighbour_sets[d][exp_type].add(s)

    melo_neighbour_counts: Dict[str, Counter] = {}
    for melo_id, by_type in neighbour_sets.items():
        melo_neighbour_counts[melo_id] = Counter({t: len(ids) for t, ids in by_type.items()})

    return IstSignature(
        ist_graph_id=ist_graph_id,
        node_counts=node_counts,
        dir_counts=dict(dir_counts),
        edge_counts=edge_counts,
        attach_ratio_by_type=attach_ratio_by_type,
        melo_neighbour_counts=melo_neighbour_counts,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _hard_bounds_ok(ist: IstSignature, tpl: TemplateSignature) -> bool:
    """Hard feasibility check: observed counts must satisfy each type's bounds."""
    for t in ("MaLo", "MeLo", "TR", "NeLo"):
        val = int(ist.node_counts.get(t, 0))
        b = tpl.bounds_by_type.get(t, Bounds(0, None))
        if val < int(b.min):
            return False
        if b.max is not None and val > int(b.max):
            return False
    return True


def _bounds_violation_penalty(ist: IstSignature, tpl: TemplateSignature) -> float:
    """Non-negative penalty to rank infeasible templates among themselves (lower is better)."""
    pen = 0.0
    for t in ("MaLo", "MeLo", "TR", "NeLo"):
        val = float(int(ist.node_counts.get(t, 0)))
        b = tpl.bounds_by_type.get(t, Bounds(0, None))
        if val < float(b.min):
            pen += float(b.min) - val
        elif b.max is not None and val > float(b.max):
            pen += val - float(b.max)
    return float(pen)


def _specificity_factor(b: Bounds) -> float:
    """Return a mild factor in [0.5, 1.0] that prefers tighter bounds."""
    if b.max is None:
        return 0.70
    width = max(0, int(b.max) - int(b.min))
    spec = 1.0 / (1.0 + float(width))
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


def _distribution_overlap(req: Counter, obs: Counter) -> float:
    """Overlap between normalised direction distributions in [0,1]."""
    req_total = float(sum(req.values()))
    obs_total = float(sum(obs.values()))
    if req_total <= 0:
        return 1.0
    if obs_total <= 0:
        return 0.0

    overlap = 0.0
    for d, rc in req.items():
        rp = float(rc) / req_total
        op = float(obs.get(d, 0)) / obs_total
        overlap += min(rp, op)
    return float(max(0.0, min(1.0, overlap)))


def _equal_count_score(ist: IstSignature, tpl: TemplateSignature) -> float:
    """Score equal-count constraints on the type-level (approximation)."""
    cons = tpl.equal_count_constraints
    if not cons:
        return 1.0

    scores: List[float] = []
    for c in cons:
        ta = c.get("a_type")
        tb = c.get("b_type")
        if ta not in ("MaLo", "MeLo", "TR", "NeLo") or tb not in ("MaLo", "MeLo", "TR", "NeLo"):
            continue
        a = int(ist.node_counts.get(ta, 0))
        b = int(ist.node_counts.get(tb, 0))
        denom = max(1, max(a, b))
        scores.append(1.0 - abs(a - b) / denom)

    return float(sum(scores) / max(1, len(scores))) if scores else 1.0


def _melo_structure_score(ist: IstSignature, tpl: TemplateSignature) -> float:
    """Score how well mandatory MeLo roles can be covered by instance MeLos.

    Uses only mandatory, unambiguous attachment requirements (reference_to_melo is a single code).
    """
    mand_roles = [r for r in tpl.mandatory_melo_roles if r]
    if not mand_roles:
        return 1.0

    inst_melos = list(ist.melo_neighbour_counts.keys())
    if not inst_melos:
        # Template requires MeLo roles, but instance has none.
        return 0.0

    # Greedy assignment (templates are small enough; avoid heavy deps)
    def role_specificity(code: str) -> int:
        req = tpl.melo_min_attach_reqs.get(code, Counter())
        return int(sum(req.values()))

    roles_sorted = sorted(mand_roles, key=role_specificity, reverse=True)

    used: set = set()
    scores: List[float] = []

    for role in roles_sorted:
        req = tpl.melo_min_attach_reqs.get(role, Counter())

        best = 0.0
        best_id: Optional[str] = None
        for mid in inst_melos:
            if mid in used:
                continue
            obs = ist.melo_neighbour_counts.get(mid, Counter())

            if not req:
                s = 1.0
            else:
                parts: List[float] = []
                for t, k in req.items():
                    if k <= 0:
                        continue
                    parts.append(min(int(obs.get(t, 0)), int(k)) / float(k))
                s = float(sum(parts) / max(1, len(parts))) if parts else 1.0

            if s > best:
                best = s
                best_id = mid

        if best_id is None:
            scores.append(0.0)
        else:
            used.add(best_id)
            scores.append(best)

    return float(sum(scores) / max(1, len(scores))) if scores else 1.0


@dataclass
class ScoreBreakdown:
    counts: float
    mandatory: float
    dirs: float
    structure: float
    edges: float
    attachments: float
    total: float


def score_pair(
    ist: IstSignature,
    tpl: TemplateSignature,
    *,
    w_counts: float = 0.50,
    w_mandatory: float = 0.22,
    w_dirs: float = 0.13,
    w_structure: float = 0.10,
    w_edges: float = 0.03,
    w_attach: float = 0.02,
) -> ScoreBreakdown:
    """Compute a transparent similarity score and a breakdown.

    IMPORTANT: min/max bounds are treated as a **hard** constraint.
    """
    if not _hard_bounds_ok(ist, tpl):
        # Hard constraint failed: return a zero score so the template is ranked behind feasible ones.
        return ScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 1) Cardinalities / bounds: within-bounds already guaranteed, so this mainly encodes specificity
    type_weights = {"MaLo": 0.45, "MeLo": 0.45, "TR": 0.05, "NeLo": 0.05}
    s_counts = 0.0
    w_sum = 0.0
    for t, wt in type_weights.items():
        b = tpl.bounds_by_type.get(t, Bounds(0, None))
        s_counts += wt * _specificity_factor(b)
        w_sum += wt
    s_counts = s_counts / max(1e-9, w_sum)

    # 2) Mandatory roles (min_occurs>=1) coverage (type+direction, counted as min occurrences)
    mand_type_weights = {"MaLo": 0.70, "MeLo": 0.25, "TR": 0.03, "NeLo": 0.02}
    s_mand = 0.0
    w_sum = 0.0
    for t, wt in mand_type_weights.items():
        req = tpl.mandatory_dir_counts.get(t, Counter())
        obs = ist.dir_counts.get(t, Counter())
        s_mand += wt * _dir_overlap_score(req, obs)
        w_sum += wt
    s_mand = s_mand / max(1e-9, w_sum)

    # 3) Direction distribution similarity (soft)
    dir_type_weights = {"MaLo": 0.75, "TR": 0.20, "MeLo": 0.05}
    s_dirs = 0.0
    w_sum = 0.0
    for t, wt in dir_type_weights.items():
        req = tpl.mandatory_dir_counts.get(t, Counter())
        obs = ist.dir_counts.get(t, Counter())
        s_dirs += wt * _distribution_overlap(req, obs)
        w_sum += wt
    s_dirs = s_dirs / max(1e-9, w_sum)

    # 4) Structure / references (where observable)
    #    - MeLo role grouping via reference_to_melo (mandatory+unambiguous only)
    #    - equal-count constraints (type-level approximation)
    s_melo = _melo_structure_score(ist, tpl)
    s_eq = _equal_count_score(ist, tpl)
    s_struct = 0.80 * s_melo + 0.20 * s_eq

    # 5) Edge sanity (only for *mandatory* endpoint types; lower weight)
    req_malo = int(sum(tpl.mandatory_dir_counts.get("MaLo", Counter()).values()))
    req_tr = int(sum(tpl.mandatory_dir_counts.get("TR", Counter()).values()))
    req_nelo = int(sum(tpl.mandatory_dir_counts.get("NeLo", Counter()).values()))
    required_edges = {"MEMA": req_malo, "METR": req_tr, "MENE": req_nelo}
    edge_scores = []
    for rel, req_cnt in required_edges.items():
        if req_cnt <= 0:
            continue
        obs_cnt = int(ist.edge_counts.get(rel, 0))
        edge_scores.append(min(obs_cnt, req_cnt) / req_cnt)
    s_edges = float(sum(edge_scores) / max(1, len(edge_scores))) if edge_scores else 1.0

    # 6) Attachment coverage (only if Ist contains the node types)
    attach_scores = []
    for t, ratio in ist.attach_ratio_by_type.items():
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
        + w_structure * s_struct
        + w_edges * s_edges
        + w_attach * s_attach
    )

    return ScoreBreakdown(
        counts=float(s_counts),
        mandatory=float(s_mand),
        dirs=float(s_dirs),
        structure=float(s_struct),
        edges=float(s_edges),
        attachments=float(s_attach),
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
    malos = [
        str(n.get("id"))
        for n in (g.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "MaLo" and n.get("id") is not None
    ]
    melos = [
        str(n.get("id"))
        for n in (g.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "MeLo" and n.get("id") is not None
    ]
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
    p.add_argument("--ist_path", type=str, default=str("data/ist_graphs_all.jsonl"))
    p.add_argument("--templates_path", type=str, default=str("data/lbs_soll_graphs.jsonl"))
    p.add_argument("--out_path", type=str, default=str("data/rule_baseline_matches.jsonl"))
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_ist", type=int, default=None)
    p.add_argument("--max_templates", type=int, default=None)

    # Default ON: try to evaluate with your labelled subset.
    p.add_argument(
        "--bndl2mc_path",
        type=str,
        default=str("data/training_data/BNDL2MC.csv"),
        help="Optional BNDL2MC.csv for evaluation (default: data/training_data/BNDL2MC.csv). Use '' to disable.",
    )

    # Where to look for raw LBS JSONs containing `_lbs_optionality`
    p.add_argument(
        "--lbs_json_dir",
        type=str,
        default=str("data/lbs_templates"),
        help="Directory containing the raw LBS JSON files with _lbs_optionality",
    )

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

    # Load optionality catalog (source of constraints)
    lbs_json_dir = _resolve_default(Path(args.lbs_json_dir))
    if not lbs_json_dir.exists() or not lbs_json_dir.is_dir():
        raise FileNotFoundError(f"lbs_json_dir not found or not a directory: {lbs_json_dir}")
    opt_catalog = load_lbs_optionality_catalog(lbs_json_dir)

    ist_graphs = _load_jsonl(ist_path, max_lines=args.max_ist)
    templates = _load_jsonl(tpl_path, max_lines=args.max_templates)
    if not templates:
        raise RuntimeError("No templates loaded.")

    # Optional labels (evaluation). Default tries to load data/BNDL2MC.csv.
    pair_to_mcid: Dict[Tuple[str, str], str] = {}
    bndl_arg = str(args.bndl2mc_path or "").strip()
    if bndl_arg:
        bndl_path = _resolve_default(Path(bndl_arg))
        if bndl_path.exists():
            pair_to_mcid = load_bndl2mc_labels(bndl_path)
            print(f"[baseline] Loaded BNDL2MC pairs: {len(pair_to_mcid)} from {bndl_path}")
        else:
            print(f"[baseline][warn] BNDL2MC.csv not found at: {bndl_path} (evaluation disabled)")

    print(
        f"[baseline] ist_graphs={len(ist_graphs)} | templates={len(templates)} | "
        f"top_k={args.top_k} | lbs_json_dir={lbs_json_dir} | optionality_codes={len(opt_catalog)}"
    )

    tpl_sigs = [build_template_signature(t, optionality_catalog=opt_catalog) for t in templates]

    # Sanity: ensure each template has optionality
    missing = [sig.template_label for sig in tpl_sigs if sig.optionality_source_path is None]
    if missing:
        # Not fatal, but should be visible.
        print(f"[baseline][warn] Missing _lbs_optionality for templates: {sorted(set(missing))}")

    # Sanity: warn if mandatory directions are only unknown for MaLo/MeLo (often indicates wrong key)
    suspicious = []
    for sig in tpl_sigs:
        for t in ("MaLo", "MeLo"):
            req = sig.mandatory_dir_counts.get(t, Counter())
            if sum(req.values()) <= 0:
                continue
            known = sum(v for k, v in req.items() if k != "unknown")
            if known == 0:
                suspicious.append(sig.template_label)
                break
    if suspicious:
        print(
            "[baseline][warn] Suspicious template directions (mandatory roles only 'unknown') for: "
            + ", ".join(sorted(set(suspicious)))
        )

    # Evaluation counters (if labels exist)
    eval_total = 0
    eval_top1 = 0
    eval_top3 = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for g in ist_graphs:
            ist_sig = build_ist_signature(g)

            scored: List[Tuple[int, float, float, int, ScoreBreakdown]] = []
            # tuple: (hard_ok_int, total, penalty, idx, breakdown)
            for i, tpl_sig in enumerate(tpl_sigs):
                bd = score_pair(ist_sig, tpl_sig)
                hard_ok = 1 if _hard_bounds_ok(ist_sig, tpl_sig) else 0
                pen = _bounds_violation_penalty(ist_sig, tpl_sig)
                scored.append((hard_ok, bd.total, pen, i, bd))

            scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
            top_k = max(1, int(args.top_k))
            top = scored[:top_k]

            top_templates = []
            for rank, (hard_ok, score, pen, idx, bd) in enumerate(top, start=1):
                tpl = templates[idx]
                tpl_sig = tpl_sigs[idx]

                # Build checklist (interpretable)
                checklist = {
                    "hard_constraints": {
                        "bounds_ok": bool(hard_ok),
                        "bounds_violation_penalty": float(pen),
                    },
                    "optionality": {
                        "source_path": tpl_sig.optionality_source_path,
                        "version": tpl_sig.optionality_version,
                        "reference_rules_n": len(tpl_sig.reference_rules),
                        "equal_count_constraints_n": len(tpl_sig.equal_count_constraints),
                        "ambiguous_attach_reqs_n": len(tpl_sig.ambiguous_attach_reqs),
                    },
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
                            "required_min": dict(tpl_sig.mandatory_dir_counts.get(t, Counter())),
                            "observed": dict(ist_sig.dir_counts.get(t, Counter())),
                        }
                        for t in ("MaLo", "MeLo", "TR", "NeLo")
                        if sum(tpl_sig.mandatory_dir_counts.get(t, Counter()).values()) > 0
                    },
                    "structure": {
                        "mandatory_melo_roles": len(tpl_sig.mandatory_melo_roles),
                        "instance_melo_nodes": len(
                            [n for n in (g.get("nodes") or []) if isinstance(n, dict) and n.get("type") == "MeLo"]
                        ),
                        "instance_melo_neighbour_counts": {
                            melo_id: {k: int(v) for k, v in cnt.items()}
                            for melo_id, cnt in list(ist_sig.melo_neighbour_counts.items())[:10]
                        },
                        "melo_min_attach_reqs": {
                            melo_code: {k: int(v) for k, v in req.items()}
                            for melo_code, req in tpl_sig.melo_min_attach_reqs.items()
                        },
                        "equal_count_constraints": [
                            {
                                **c,
                                "obs_a": int(ist_sig.node_counts.get(c.get("a_type", ""), 0)),
                                "obs_b": int(ist_sig.node_counts.get(c.get("b_type", ""), 0)),
                            }
                            for c in tpl_sig.equal_count_constraints
                        ],
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
                            "structure": bd.structure,
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
