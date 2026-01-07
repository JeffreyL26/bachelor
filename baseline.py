"""
Baseline Matcher (Mode B): nutzt Tabellen-Semantik aus _lbs_optionality als Source-of-Truth.

Ziel:
- Ranking von LBS-Templates (Soll) für gegebene Ist-Graphen (SAP-Export-Graphen).
- Kein Training, deterministisch, erklärt sich über struktur-/attributbasierten Score.
- Optionalität/Min/Max wird NICHT über T_VAR.T_ROLE.FREQUENCY_* interpretiert,
  sondern über _lbs_optionality.lbs_objects.

Erwartetes Graph-Format (Ist wie Soll-intern):
{
  "graph_id": "...",
  "nodes": [{"id": "...", "type": "MaLo|MeLo|TR|NeLo", "attrs": {...}}, ...],
  "edges": [{"src": "...", "dst": "...", "rel": "MEMA|METR|MEME|MENE"}, ...],
  "graph_attrs": {"pattern": "MaLo:MeLo", "malo_count": int, "melo_count": int}
}

Hinweise:
- Viele Ist-Graphen haben wenige Attribute (z.B. MeLo ohne 'function'). Deshalb ist
  Struktur (Nachbarschaften/Relationstypen) die primäre Signalquelle.
- Flexible zusätzliche TR (max_occurs = "N") werden nur schwach bestraft.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonDict = Dict[str, Any]


# -----------------------------
# Helpers: IO
# -----------------------------
def load_json(path: str) -> JsonDict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_graphs(path: str, limit: Optional[int] = None) -> List[JsonDict]:
    graphs: List[JsonDict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            graphs.append(json.loads(line))
            if limit is not None and len(graphs) >= limit:
                break
    return graphs


def write_jsonl(path: str, rows: Iterable[JsonDict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Template parsing (Ver2 JSON)
# -----------------------------
def _descr_de(role_row: JsonDict) -> str:
    for d in role_row.get("T_DESCR", []) or []:
        if (d.get("LANG") or "").upper() == "D":
            return d.get("DESCR") or ""
    # fallback: first
    t = role_row.get("T_DESCR", [])
    if t:
        return t[0].get("DESCR") or ""
    return ""


def _core_type_and_id(role_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristik wie in graph_templates.py:
    Rolle kann z.B. 'AME_BZG' sein; Kernrolle ist 'ME_BZG'.
    Wir mappen auf (ntype, core_id).

    Erwartete Präfixe im Template: MA_, ME_, TR_, NELO_ (oder ähnliche).
    Außerdem existieren oft Anlage-/Hilfsrollen: AMA_, AME_, ANE_, ATR_...
    """
    rid = (role_id or "").strip()
    if not rid:
        return None, None

    # Normalisiere Anlage-/Hilfspräfixe: AMA_ -> MA_, AME_ -> ME_, ANE_ -> NELO_, ATR_ -> TR_
    # (nicht garantiert, aber in deinen bisherigen Ver2-Dateien üblich)
    rid_norm = rid
    rid_norm = re.sub(r"^AMA_", "MA_", rid_norm)
    rid_norm = re.sub(r"^AME_", "ME_", rid_norm)
    rid_norm = re.sub(r"^ANE_", "NELO_", rid_norm)
    rid_norm = re.sub(r"^ATR_", "TR_", rid_norm)

    if rid_norm.startswith("MA_"):
        return "MaLo", rid_norm
    if rid_norm.startswith("ME_"):
        return "MeLo", rid_norm
    if rid_norm.startswith("TR_"):
        return "TR", rid_norm
    if rid_norm.startswith("NELO_"):
        return "NeLo", rid_norm

    # Manche Dateien nutzen MELO_/MALO_ etc.
    if rid_norm.startswith("MALO_"):
        return "MaLo", rid_norm
    if rid_norm.startswith("MELO_"):
        return "MeLo", rid_norm

    return None, None


def _extract_rel_type(rr: JsonDict, t1: str, t2: str) -> str:
    rel_type = None
    for att in rr.get("T_OBJRAT", []) or []:
        if att.get("ATTR_CATEGORY") == "REL_TYPE":
            v = (att.get("ATTR_VALUE") or "").strip()
            if v:
                rel_type = v
                break
    if rel_type:
        return rel_type

    # Fallbacks (wie graph_templates.py)
    if t1 == "MeLo" and t2 == "MaLo":
        return "MEMA"
    if t1 == "MeLo" and t2 == "NeLo":
        return "MENE"
    if (t1 == "MeLo" and t2 == "TR") or (t1 == "TR" and t2 == "MeLo"):
        return "METR"
    if t1 == "MeLo" and t2 == "MeLo":
        return "MEME"
    return "REL"


def _max_to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    s = str(x).strip()
    if not s:
        return None
    if s.upper() == "N":
        return None  # unbounded
    try:
        return int(s)
    except ValueError:
        return None


def _parse_direction_hint(direction_hint: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    'consumption+generation (Netzübergabe)' -> ('consumption+generation', 'Netzübergabe')
    """
    if not direction_hint:
        return None, None
    s = direction_hint.strip()
    if "(" in s and s.endswith(")"):
        base, ctx = s.split("(", 1)
        return base.strip(), ctx[:-1].strip()
    return s, None


def _meta_for_object(obj: JsonDict) -> JsonDict:
    """Normalisiere Metadaten in ein kompaktes Feature-Set für Matching."""
    otype = obj.get("object_type")
    level = obj.get("level")
    min_occ = obj.get("min_occurs", 0)
    max_occ = obj.get("max_occurs", "N")
    flex = obj.get("flexibility")

    # direction field variants
    direction = obj.get("direction")
    tr_dir = obj.get("tr_direction")
    dir_hint = obj.get("direction_hint")

    direction_base = None
    ctx = None
    if isinstance(dir_hint, str):
        direction_base, ctx = _parse_direction_hint(dir_hint)
    elif isinstance(direction, str):
        direction_base = direction.strip()
    elif isinstance(tr_dir, str):
        direction_base = tr_dir.strip()

    fn = obj.get("melo_function")

    return {
        "object_type": otype,
        "level": level,
        "min_occurs": min_occ,
        "max_occurs": max_occ,
        "flexibility": flex,
        "direction": direction_base,
        "function": fn,
        "direction_ctx": ctx,
    }


def parse_ver2_template(path: str, include_all_variants: bool = True) -> Tuple[List[JsonDict], List[str]]:
    """
    Returns: (template_graphs, warnings)

    template_graph = {
      "template_id": "...",
      "lbs_code": "...",
      "variant_id": "...",
      "nodes": [...],
      "edges": [...],
      "graph_attrs": {...},
      "template_meta": { ... }  # allow_extra_by_type etc.
    }
    """
    warnings: List[str] = []
    lbs = load_json(path)

    mc_list = lbs.get("T_MC", [])
    if not mc_list:
        raise ValueError(f"{os.path.basename(path)}: missing T_MC")
    mc = mc_list[0]
    lbs_code = mc.get("LBS_CODE")
    if lbs_code is None:
        # fallback: _lbs_optionality.lbs_code
        lbs_code = lbs.get("_lbs_optionality", {}).get("lbs_code")
    lbs_code_str = str(lbs_code) if lbs_code is not None else "UNKNOWN"

    # role meta table
    role_meta: Dict[str, JsonDict] = {}
    for r in lbs.get("T_ROLE", []) or []:
        rid = (r.get("ID") or "").strip()
        role_meta[rid] = {
            "id": rid,
            "descr": _descr_de(r),
            "level": int(r.get("LBS_OBJECT_LEVEL", 0) or 0),
            "code": r.get("LBS_OBJECT_CODE", 0) or 0,
            "attrs": {a.get("ATTR_CATEGORY"): a.get("ATTR_VALUE") for a in (r.get("T_ATTR", []) or [])},
        }

    # semantics meta (_lbs_optionality)
    lbs_opt = lbs.get("_lbs_optionality", {}) or {}
    lbs_objects = lbs_opt.get("lbs_objects", []) or []
    obj_by_code: Dict[str, JsonDict] = {}
    for o in lbs_objects:
        oc = str(o.get("object_code") or "").strip()
        if oc:
            obj_by_code[oc] = _meta_for_object(o)

    # allow-extra flags by type (Mode B)
    allow_extra_by_type: Dict[str, bool] = {"MaLo": False, "MeLo": False, "TR": False, "NeLo": False}
    for o in lbs_objects:
        t = o.get("object_type")
        if t not in allow_extra_by_type:
            continue
        mx = o.get("max_occurs", None)
        mx_i = _max_to_int(mx)
        if mx is None:
            continue
        if isinstance(mx, str) and mx.upper() == "N":
            allow_extra_by_type[t] = True
        elif mx_i is not None and mx_i > 1:
            allow_extra_by_type[t] = True

    # variants
    var_list = mc.get("T_VAR", []) or []
    if not var_list:
        raise ValueError(f"{os.path.basename(path)}: missing T_VAR")

    chosen_vars = var_list if include_all_variants else [var_list[0]]

    template_graphs: List[JsonDict] = []
    for var in chosen_vars:
        var_id = var.get("MCVARID") or var.get("PRIORITY") or "VAR"
        var_id = str(var_id)

        # Nodes: one node per core_id (like graph_templates.py)
        nodes_by_id: Dict[str, JsonDict] = {}
        for rr in var.get("T_ROLE", []) or []:
            role_id = (rr.get("ROLE_ID") or "").strip()
            ntype, core_id = _core_type_and_id(role_id)
            if not ntype or not core_id:
                continue

            meta = role_meta.get(core_id, role_meta.get(role_id, {}))
            level = int(meta.get("level", 0) or 0)
            code = meta.get("code", 0) or 0

            # ignore helper roles if no object code & no level
            if level == 0 and code == 0:
                continue

            if core_id not in nodes_by_id:
                attrs: Dict[str, Any] = {"level": level, "object_code": str(code) if code else None}
                # Mode B: attach canonical semantics if available via _lbs_optionality
                if code and str(code) in obj_by_code:
                    om = obj_by_code[str(code)]
                    # merge canonical semantics
                    for k in ("level", "direction", "function"):
                        if om.get(k) is not None:
                            attrs[k] = om.get(k)
                    attrs["min_occurs"] = om.get("min_occurs", 0)
                    attrs["max_occurs"] = om.get("max_occurs", "N")
                    attrs["flexibility"] = om.get("flexibility")
                    attrs["direction_ctx"] = om.get("direction_ctx")
                else:
                    # Unknown semantics -> do NOT mark required (avoid false penalties)
                    attrs["min_occurs"] = 0
                    attrs["max_occurs"] = "N"

                nodes_by_id[core_id] = {"id": core_id, "type": ntype, "attrs": attrs}

        nodes = list(nodes_by_id.values())

        # Edges
        edges: List[JsonDict] = []
        for rr in var.get("T_ROLEREL", []) or []:
            o1 = (rr.get("OBJECT_ID_1") or "").strip()
            o2 = (rr.get("OBJECT_ID_2") or "").strip()
            t1, id1 = _core_type_and_id(o1)
            t2, id2 = _core_type_and_id(o2)
            if not t1 or not t2 or not id1 or not id2:
                continue
            if id1 not in nodes_by_id or id2 not in nodes_by_id:
                # ignore edges to helper roles
                continue
            rel = _extract_rel_type(rr, t1, t2)
            edges.append({"src": id1, "dst": id2, "rel": rel})

        malo_count = sum(1 for n in nodes if n["type"] == "MaLo")
        melo_count = sum(1 for n in nodes if n["type"] == "MeLo")
        pattern = f"{malo_count}:{melo_count}"

        template_graphs.append({
            "template_id": f"{lbs_code_str}::{var_id}::{os.path.basename(path)}",
            "lbs_code": lbs_code_str,
            "variant_id": var_id,
            "nodes": nodes,
            "edges": edges,
            "graph_attrs": {"pattern": pattern, "malo_count": malo_count, "melo_count": melo_count},
            "template_meta": {
                "allow_extra_by_type": allow_extra_by_type,
                "source_file": os.path.basename(path),
            }
        })

    if not lbs_objects:
        warnings.append(f"{os.path.basename(path)}: missing _lbs_optionality.lbs_objects (Mode B will degrade to min_occurs=0 for unknown nodes).")

    return template_graphs, warnings


def load_templates_from_dir(templates_dir: str, include_all_variants: bool = True) -> Tuple[List[JsonDict], List[str]]:
    """
    Loads all *.json in templates_dir as Ver2 templates. Handles URL-encoded duplicates:
    - prefer a decoded filename if both exist and are identical lbs_code.
    """
    warnings: List[str] = []
    files = [os.path.join(templates_dir, f) for f in os.listdir(templates_dir) if f.lower().endswith(".json")]
    files.sort()

    # Dedup heuristic: if both '9992 00000 012 5 - Ver2.json' and '9992%2000000%20012%205%20-%20Ver2.json' exist,
    # prefer the decoded one.
    preferred: Dict[str, str] = {}
    for p in files:
        base = os.path.basename(p)
        decoded = re.sub(r"%20", " ", base)
        key = decoded  # approximate key
        if key not in preferred:
            preferred[key] = p
        else:
            # if we already have decoded and current is encoded, keep decoded
            if "%20" in os.path.basename(preferred[key]) and "%20" not in base:
                preferred[key] = p

    chosen_files = sorted(set(preferred.values()))
    templates: List[JsonDict] = []
    for p in chosen_files:
        try:
            tgs, w = parse_ver2_template(p, include_all_variants=include_all_variants)
            templates.extend(tgs)
            warnings.extend(w)
        except Exception as e:
            warnings.append(f"{os.path.basename(p)}: failed to parse ({e})")
    return templates, warnings


# -----------------------------
# Baseline scoring (Mode B)
# -----------------------------
@dataclass
class BaselineConfig:
    top_k: int = 5
    filter_by_pattern: bool = True
    # scoring weights
    w_attr: float = 0.35          # attribute vs structure similarity
    alpha_node: float = 0.70      # node-term vs edge-term
    beta_required: float = 0.80   # penalty weight for missing required template nodes
    extra_penalty_strict: float = 1.0
    extra_penalty_flexible: float = 0.25
    # edge matching robustness
    edge_allow_reverse: bool = True


def _index_nodes(graph: JsonDict) -> Tuple[List[JsonDict], Dict[str, int]]:
    nodes = graph.get("nodes", []) or []
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    return nodes, idx


def _edge_index(graph: JsonDict) -> Dict[Tuple[int, int, str], int]:
    """
    Build multiset index on (src_idx, dst_idx, rel).
    """
    nodes, node_idx = _index_nodes(graph)
    eidx: Dict[Tuple[int, int, str], int] = {}
    for e in graph.get("edges", []) or []:
        if e.get("src") not in node_idx or e.get("dst") not in node_idx:
            continue
        s = node_idx[e["src"]]
        t = node_idx[e["dst"]]
        rel = e.get("rel") or "REL"
        key = (s, t, rel)
        eidx[key] = eidx.get(key, 0) + 1
    return eidx


def _struct_signature(graph: JsonDict) -> List[Dict[str, int]]:
    """
    For each node: signature dict counts of incident relation types and neighbor types.
    Uses both directions.
    """
    nodes, node_idx = _index_nodes(graph)
    sigs: List[Dict[str, int]] = [dict() for _ in nodes]

    def bump(i: int, k: str, v: int = 1) -> None:
        sigs[i][k] = sigs[i].get(k, 0) + v

    for e in graph.get("edges", []) or []:
        src = e.get("src")
        dst = e.get("dst")
        rel = e.get("rel") or "REL"
        if src not in node_idx or dst not in node_idx:
            continue
        si = node_idx[src]
        di = node_idx[dst]
        bump(si, f"rel::{rel}", 1)
        bump(di, f"rel::{rel}", 1)

        t_dst = nodes[di].get("type") or "UNK"
        t_src = nodes[si].get("type") or "UNK"
        bump(si, f"neigh::{t_dst}", 1)
        bump(di, f"neigh::{t_src}", 1)

    for i in range(len(nodes)):
        deg = 0
        for k, v in sigs[i].items():
            if k.startswith("rel::"):
                deg += v
        bump(i, "deg_total", deg)
    return sigs


def _l1_distance(d1: Dict[str, int], d2: Dict[str, int]) -> int:
    keys = set(d1.keys()) | set(d2.keys())
    return sum(abs(d1.get(k, 0) - d2.get(k, 0)) for k in keys)


def _attr_similarity(a: JsonDict, b: JsonDict) -> float:
    """
    Attribute similarity in [0,1]. Compares only shared, non-null attributes.
    Missing attributes do not penalize.
    """
    keys = ["type", "direction", "function", "level", "voltage_level"]
    hits = 0
    checks = 0

    # types must match (hard), but keep it also in similarity for stability
    if (a.get("type") or "") != (b.get("type") or ""):
        return 0.0

    aa = a.get("attrs", {}) or {}
    bb = b.get("attrs", {}) or {}

    for k in keys[1:]:
        va = aa.get(k)
        vb = bb.get(k)
        if va is None or vb is None:
            continue
        checks += 1
        if str(va) == str(vb):
            hits += 1

    if checks == 0:
        return 0.5  # unknown -> neutral
    return hits / checks


def _node_similarity_matrix(src: JsonDict, tgt: JsonDict, cfg: BaselineConfig) -> List[List[float]]:
    s_nodes, _ = _index_nodes(src)
    t_nodes, _ = _index_nodes(tgt)
    s_sig = _struct_signature(src)
    t_sig = _struct_signature(tgt)

    mat: List[List[float]] = [[0.0 for _ in t_nodes] for _ in s_nodes]
    for i, sn in enumerate(s_nodes):
        for j, tn in enumerate(t_nodes):
            if sn.get("type") != tn.get("type"):
                mat[i][j] = -1.0  # incompatible
                continue
            attr_sim = _attr_similarity(sn, tn)

            # structure sim: 1 / (1 + L1)
            dist = _l1_distance(s_sig[i], t_sig[j])
            struct_sim = 1.0 / (1.0 + float(dist))

            mat[i][j] = cfg.w_attr * attr_sim + (1.0 - cfg.w_attr) * struct_sim
    return mat


def _greedy_assignment(sim: List[List[float]]) -> List[Tuple[int, Optional[int], float]]:
    """
    Greedy 1:1 assignment maximizing similarity. Returns list of (src_i, tgt_j_or_None, sim).
    """
    if not sim:
        return []
    ns = len(sim)
    nt = len(sim[0]) if ns else 0

    pairs: List[Tuple[float, int, int]] = []
    for i in range(ns):
        for j in range(nt):
            v = sim[i][j]
            if v < 0:
                continue
            pairs.append((v, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_s = set()
    used_t = set()
    match: Dict[int, Tuple[int, float]] = {}

    for v, i, j in pairs:
        if i in used_s or j in used_t:
            continue
        used_s.add(i)
        used_t.add(j)
        match[i] = (j, v)

    out: List[Tuple[int, Optional[int], float]] = []
    for i in range(ns):
        if i in match:
            j, v = match[i]
            out.append((i, j, v))
        else:
            out.append((i, None, 0.0))
    return out


def _edge_consistency(src: JsonDict, tgt: JsonDict, mapping: List[Tuple[int, Optional[int], float]], cfg: BaselineConfig) -> float:
    """
    Fraction of src edges between mapped nodes that exist in tgt with same rel.
    """
    s_nodes, s_idx = _index_nodes(src)
    t_nodes, t_idx = _index_nodes(tgt)
    if not s_nodes or not t_nodes:
        return 0.0

    s_e = src.get("edges", []) or []
    t_eidx = _edge_index(tgt)

    # build src_idx -> tgt_idx mapping
    map_st: Dict[int, int] = {}
    for si, tj, _ in mapping:
        if tj is not None:
            map_st[si] = tj

    total = 0
    ok = 0
    for e in s_e:
        if e.get("src") not in s_idx or e.get("dst") not in s_idx:
            continue
        si = s_idx[e["src"]]
        di = s_idx[e["dst"]]
        if si not in map_st or di not in map_st:
            continue  # only evaluate edges between mapped nodes
        total += 1
        ts = map_st[si]
        td = map_st[di]
        rel = e.get("rel") or "REL"
        if (ts, td, rel) in t_eidx:
            ok += 1
        elif cfg.edge_allow_reverse and (td, ts, rel) in t_eidx:
            ok += 1

    if total == 0:
        return 0.0
    return ok / total


def _required_template_penalty(tgt: JsonDict, mapping: List[Tuple[int, Optional[int], float]]) -> float:
    """
    Penalize missing required template nodes (min_occurs>=1).
    Required info comes from Mode B (attached to template nodes).
    """
    t_nodes, _ = _index_nodes(tgt)
    if not t_nodes:
        return 0.0

    mapped_t = set(tj for _, tj, _ in mapping if tj is not None)

    req_total = 0
    req_missed = 0
    for j, n in enumerate(t_nodes):
        attrs = n.get("attrs", {}) or {}
        min_occ = attrs.get("min_occurs", 0)
        if isinstance(min_occ, str):
            try:
                min_occ = int(min_occ)
            except ValueError:
                min_occ = 0
        if min_occ >= 1:
            req_total += 1
            if j not in mapped_t:
                req_missed += 1

    if req_total == 0:
        return 0.0
    return req_missed / req_total


def _extra_src_penalty(src: JsonDict, tgt: JsonDict, mapping: List[Tuple[int, Optional[int], float]], cfg: BaselineConfig) -> float:
    """
    Penalize unmatched src nodes; reduce penalty for types where template allows extra (max_occurs='N' or >1).
    Uses template_meta.allow_extra_by_type (Mode B).
    """
    s_nodes, _ = _index_nodes(src)
    allow_extra = (tgt.get("template_meta") or {}).get("allow_extra_by_type") or {}
    unmatched = [si for si, tj, _ in mapping if tj is None]
    if not unmatched:
        return 0.0

    total_w = 0.0
    for si in unmatched:
        t = s_nodes[si].get("type") or "UNK"
        if allow_extra.get(t, False):
            total_w += cfg.extra_penalty_flexible
        else:
            total_w += cfg.extra_penalty_strict
    # normalize later by max(|Vsrc|,|Vtgt|)
    return total_w


def score_pair_mode_b(src: JsonDict, tgt: JsonDict, cfg: BaselineConfig) -> Tuple[float, JsonDict]:
    s_nodes, _ = _index_nodes(src)
    t_nodes, _ = _index_nodes(tgt)
    ns = len(s_nodes)
    nt = len(t_nodes)
    denom = max(ns, nt, 1)

    sim = _node_similarity_matrix(src, tgt, cfg)
    mapping = _greedy_assignment(sim)

    node_sum = sum(v for _, tj, v in mapping if tj is not None)
    extra_pen = _extra_src_penalty(src, tgt, mapping, cfg)
    required_pen = _required_template_penalty(tgt, mapping)
    edge_score = _edge_consistency(src, tgt, mapping, cfg)

    node_term = (node_sum - extra_pen) / denom
    score = cfg.alpha_node * node_term + (1.0 - cfg.alpha_node) * edge_score - cfg.beta_required * required_pen

    # provide mapping in ID space for interpretability
    s_nodes, _ = _index_nodes(src)
    t_nodes, _ = _index_nodes(tgt)
    mapping_out: List[JsonDict] = []
    for si, tj, v in mapping:
        mapping_out.append({
            "src_index": si,
            "src_id": s_nodes[si]["id"] if si < len(s_nodes) else None,
            "src_type": s_nodes[si].get("type"),
            "tgt_index": tj,
            "tgt_id": t_nodes[tj]["id"] if (tj is not None and tj < len(t_nodes)) else None,
            "tgt_type": t_nodes[tj].get("type") if tj is not None else None,
            "sim": v,
        })

    detail = {
        "node_sum": node_sum,
        "extra_pen": extra_pen,
        "required_pen": required_pen,
        "edge_score": edge_score,
        "denom": denom,
        "node_term": node_term,
        "alpha_node": cfg.alpha_node,
        "beta_required": cfg.beta_required,
        "w_attr": cfg.w_attr,
        "allow_extra_by_type": (tgt.get("template_meta") or {}).get("allow_extra_by_type"),
    }
    return score, {"mapping": mapping_out, "detail": detail}


def _pattern_from_graph(g: JsonDict) -> str:
    ga = g.get("graph_attrs") or {}
    pat = ga.get("pattern")
    if pat:
        return str(pat)
    # fallback from counts
    nodes = g.get("nodes", []) or []
    malo = sum(1 for n in nodes if n.get("type") == "MaLo")
    melo = sum(1 for n in nodes if n.get("type") == "MeLo")
    return f"{malo}:{melo}"


def rank_templates_for_graph(src: JsonDict, templates: List[JsonDict], cfg: BaselineConfig) -> List[Tuple[float, JsonDict, JsonDict]]:
    src_pat = _pattern_from_graph(src)

    candidates = templates
    if cfg.filter_by_pattern:
        same = [t for t in templates if _pattern_from_graph(t) == src_pat]
        candidates = same if same else templates

    scored: List[Tuple[float, JsonDict, JsonDict]] = []
    for t in candidates:
        s, pack = score_pair_mode_b(src, t, cfg)
        scored.append((s, t, pack))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[: cfg.top_k]


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline Matcher (Mode B) using _lbs_optionality semantics.")
    ap.add_argument("--ist", required=True, help="Path to JSONL with Ist-Graphen.")
    ap.add_argument("--templates_dir", required=True, help="Directory with Ver2 template JSON files.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--top_k", type=int, default=5)

    ap.add_argument("--no_pattern_filter", action="store_true", help="Disable pattern pre-filter.")
    ap.add_argument("--include_all_variants", action="store_true", help="Load all T_VAR variants per template.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of Ist graphs.")

    ap.add_argument("--w_attr", type=float, default=0.35)
    ap.add_argument("--alpha_node", type=float, default=0.70)
    ap.add_argument("--beta_required", type=float, default=0.80)
    ap.add_argument("--extra_strict", type=float, default=1.0)
    ap.add_argument("--extra_flexible", type=float, default=0.25)
    ap.add_argument("--edge_no_reverse", action="store_true", help="Do not accept reversed edges in edge consistency.")

    args = ap.parse_args()

    cfg = BaselineConfig(
        top_k=args.top_k,
        filter_by_pattern=not args.no_pattern_filter,
        w_attr=args.w_attr,
        alpha_node=args.alpha_node,
        beta_required=args.beta_required,
        extra_penalty_strict=args.extra_strict,
        extra_penalty_flexible=args.extra_flexible,
        edge_allow_reverse=not args.edge_no_reverse,
    )

    templates, warnings = load_templates_from_dir(args.templates_dir, include_all_variants=args.include_all_variants)

    if warnings:
        print("WARNINGS while loading templates:")
        for w in warnings:
            print(" -", w)

    ist_graphs = load_jsonl_graphs(args.ist, limit=args.limit)

    out_rows: List[JsonDict] = []
    for g in ist_graphs:
        gid = g.get("graph_id")
        pat = _pattern_from_graph(g)
        ranked = rank_templates_for_graph(g, templates, cfg)

        if not ranked:
            out_rows.append({
                "ist_graph_id": gid,
                "ist_pattern": pat,
                "top_k": [],
                "best": None,
            })
            continue

        top_list = []
        for score, t, pack in ranked:
            top_list.append({
                "template_id": t.get("template_id"),
                "lbs_code": t.get("lbs_code"),
                "variant_id": t.get("variant_id"),
                "template_pattern": _pattern_from_graph(t),
                "score": score,
                "mapping": pack["mapping"],
                "baseline_detail": pack["detail"],
                "template_source_file": (t.get("template_meta") or {}).get("source_file"),
            })

        best = top_list[0]
        out_rows.append({
            "ist_graph_id": gid,
            "ist_pattern": pat,
            "top_k": top_list,
            "best": {
                "template_id": best["template_id"],
                "lbs_code": best["lbs_code"],
                "variant_id": best["variant_id"],
                "score": best["score"],
                "template_pattern": best["template_pattern"],
                "template_source_file": best["template_source_file"],
            },
        })

    write_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} results to {args.out}")


if __name__ == "__main__":
    main()
