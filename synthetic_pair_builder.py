
from __future__ import annotations

import copy
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

# JSON-Graph und Trainingspaar
TGraph = Dict[str, Any]
TPair = Dict[str, Any]


# =============================================================================
# IO: Templates laden und permutieren
# =============================================================================

def load_templates_jsonl(path: str) -> List[TGraph]:
    """
    Lädt normalisierte Template-Graphen aus einer JSONL-Datei.

    Erwartetes Format pro Zeile:
      {"graph_id": ..., "label": ..., "nodes": [...], "edges": [...], "graph_attrs": {...}}
    """
    graphs: List[TGraph] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            graphs.append(json.loads(line))
    return graphs


def permute_graph(g: TGraph) -> Tuple[TGraph, List[int]]:
    """
    Erzeugt eine permutierte Kopie eines Graphen:
      - die Knotenreihenfolge wird zufällig vertauscht
      - Knoten-IDs bleiben gleich
      - Kanten bleiben über (src/dst)-IDs definiert

    Zusätzlich wird eine Permutationsliste zurückgegeben:
      perm[i] = Index des Knoten i (im Original) in der neuen Knotenliste.

    Hinweis:
      - Deep-Copy, damit spätere Augmentierungen das Original nicht mutieren.
    """
    nodes = list(g.get("nodes") or [])
    n = len(nodes)

    order = list(range(n))
    random.shuffle(order)

    new_nodes = [copy.deepcopy(nodes[i]) for i in order]
    id_to_new_idx = {node.get("id"): idx for idx, node in enumerate(new_nodes) if node.get("id") is not None}

    perm: List[int] = []
    for i in range(n):
        nid = nodes[i].get("id")
        perm.append(id_to_new_idx.get(nid, -1))

    new_g: TGraph = {
        "graph_id": f'{g.get("graph_id", "")}|perm',
        "label": g.get("label"),
        "nodes": new_nodes,
        "edges": copy.deepcopy(list(g.get("edges", []))),
        "graph_attrs": copy.deepcopy(dict(g.get("graph_attrs", {}))),
    }
    return new_g, perm


# =============================================================================
# Utils: Optionals / Constraints aus Template-Attrs (min/max/optional/flexibility)
# =============================================================================

def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_unbounded_max(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().upper() == "N":
        return True
    return False


def _node_optionality(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrahiert optionale Metainfos aus node["attrs"] (falls vorhanden).
    Diese Keys existieren in deinen neuen Soll-Graphen (_lbs_optionality):
      - min_occurs (int)
      - max_occurs (int oder "N")
      - flexibility ("starr"/"flexibel")
      - optional (bool)
    """
    attrs = node.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    min_occurs = attrs.get("min_occurs")
    max_occurs = attrs.get("max_occurs")
    optional_flag = attrs.get("optional")

    is_optional = bool(optional_flag) if optional_flag is not None else (_as_int(min_occurs, 0) == 0)

    flexibility = attrs.get("flexibility")
    flex = str(flexibility).lower().strip() if isinstance(flexibility, str) else ""

    max_int = None
    if _is_unbounded_max(max_occurs):
        max_int = None
    else:
        # int oder int-String
        if isinstance(max_occurs, int):
            max_int = max_occurs
        elif isinstance(max_occurs, str) and max_occurs.strip().isdigit():
            max_int = int(max_occurs.strip())
        else:
            max_int = None

    return {
        "is_optional": is_optional,
        "min_occurs": _as_int(min_occurs, 0),
        "max_occurs": max_int,          # None = unbounded/unknown
        "max_is_unbounded": _is_unbounded_max(max_occurs),
        "is_flexible": flex.startswith("flex"),
    }


def _edge_key(e: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(e.get("src")), str(e.get("dst")), str(e.get("rel")))


def _recount_types(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"MaLo": 0, "MeLo": 0, "TR": 0, "NeLo": 0}
    for n in nodes:
        t = n.get("type")
        if t in counts:
            counts[t] += 1
    return counts


def _sync_graph_attrs_counts(g: TGraph) -> None:
    """
    Hält optionale Zählwerte in graph_attrs konsistent, wenn Knoten entfernt/ergänzt wurden.

    Wichtig:
      - In den neuen Soll-Graphen gibt es typischerweise *_min/_max und *_node_types,
        NICHT die alten *_count Keys.
      - Diese Funktion updatet daher nur *_count Keys (falls vorhanden), um ältere
        Downstream-Skripte nicht zu brechen. *_min/_max bleiben bewusst unverändert
        (sie sind Template-Constraints und keine Instanz-Zählwerte).
    """
    ga = g.get("graph_attrs")
    if not isinstance(ga, dict):
        return

    nodes = g.get("nodes", [])
    if not isinstance(nodes, list):
        return

    counts = _recount_types(nodes)

    if "malo_count" in ga:
        ga["malo_count"] = counts["MaLo"]
    if "melo_count" in ga:
        ga["melo_count"] = counts["MeLo"]
    if "tr_count" in ga:
        ga["tr_count"] = counts["TR"]
    if "nelo_count" in ga:
        ga["nelo_count"] = counts["NeLo"]



# =============================================================================
# Optional: Template-Meta aus Graph-B entfernen (macht B "Ist-näher")
# =============================================================================

# In den Soll-Graphen (aus _lbs_optionality) stehen zusätzliche Meta-Keys in node["attrs"],
# die in echten Ist-Graphen typischerweise NICHT vorkommen (min/max/optional/flexibility/...).
#
# DGMC sieht diese Keys aktuell nicht (graph_pipeline encodiert nur Typ + Richtung),
# aber:
#   (a) es ist realistischer, wenn Graph-B diese Meta-Infos nicht trägt
#   (b) es verhindert "Leakage", falls du später weitere Attrs als Features addierst.

TEMPLATE_META_ATTR_KEYS: set[str] = {
    "min_occurs",
    "max_occurs",
    "flexibility",
    "optional",
    "object_code",
    "level",
    "reference_to_melo",
}

def strip_template_meta_attrs(
    g: TGraph,
    keys: set[str] = TEMPLATE_META_ATTR_KEYS,
) -> int:
    """
    Entfernt Template-spezifische Meta-Attribute aus allen Nodes eines Graphen (in-place).

    :return: Anzahl entfernter (key,value)-Paare
    """
    removed = 0
    for node in g.get("nodes", []) or []:
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue
        for k in list(attrs.keys()):
            if k in keys:
                del attrs[k]
                removed += 1
    return removed


# =============================================================================
# Phase A: Attribute-/Edge-Augmentation (Ist-Noise)
# =============================================================================

# Attribute-Dropout: fehlende Attribute simulieren.
# Wichtig: An aktuelle graph_pipeline angepasst (direction + Fallback-Keys).
DEFAULT_ATTR_DROPOUT_BY_TYPE: Dict[str, Dict[str, float]] = {
    # Für MaLo/MeLo/TR ist "direction" momentan der zentrale Vergleichspunkt in deiner Pipeline.
    "MaLo": {"direction": 0.25},
    "MeLo": {"direction": 0.20, "function": 0.15, "dynamic": 0.15},
    # TR: direction kann auch via tr_type_code/art_der_technischen_ressource kommen (Fallback in Pipeline)
    "TR":   {"direction": 0.25, "tr_type_code": 0.10, "art_der_technischen_ressource": 0.10},
    "NeLo": {},
}

# Edge-Dropout: METR häufiger droppen (TR-Beziehungen sind in Ist häufiger unvollständig)
DEFAULT_EDGE_DROPOUT_BY_REL: Dict[str, float] = {
    "MEMA": 0.10,
    "METR": 0.30,
    "MENE": 0.20,
    "MEME": 0.10,
    "*": 0.15,  # fallback für unbekannte rels
}


def apply_attribute_dropout(
    g: TGraph,
    probs_by_type: Dict[str, Dict[str, float]] = DEFAULT_ATTR_DROPOUT_BY_TYPE
) -> int:
    """
    Entfernt ausgewählte Attribute aus node["attrs"], um fehlende Information zu simulieren.
    :return: Anzahl gelöschter Attribute
    """
    dropped = 0
    for node in g.get("nodes", []):
        ntype = node.get("type")
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue
        probs = probs_by_type.get(str(ntype), {})
        for key, p in probs.items():
            if key in attrs and random.random() < float(p):
                del attrs[key]
                dropped += 1
    return dropped


def apply_edge_dropout(
    g: TGraph,
    drop_by_rel: Dict[str, float] = DEFAULT_EDGE_DROPOUT_BY_REL
) -> int:
    """
    Entfernt zufällig Kanten aus g["edges"].
    :return: Anzahl gedroppter Kanten
    """
    edges = list(g.get("edges", []))
    if not edges:
        return 0

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        rel = e.get("rel") or e.get("type") or e.get("edge_type")
        rel = str(rel).strip().upper() if rel is not None else None
        p = float(drop_by_rel.get(rel, drop_by_rel.get("*", 0.0)))
        if p > 0.0 and random.random() < p:
            dropped += 1
            continue
        # normalisiere auf "rel", damit downstream konsistent ist
        e2 = dict(e)
        if rel is not None:
            e2["rel"] = rel
        kept.append(e2)

    g["edges"] = kept
    return dropped


def apply_edge_less(g: TGraph) -> int:
    """
    Entfernt alle Kanten (edge-less Variante).
    :return: Anzahl entfernter Kanten
    """
    n = len(g.get("edges", []))
    g["edges"] = []
    return n


# =============================================================================
# Phase B: Partial Matching (Node-Varianz)
# =============================================================================

# Basis-Node-Dropout nach Typ (wird durch Optionality weiter skaliert).
# Hinweis: In der aktuellen Thesis-Pipeline sind MaLo/MeLo "strukturell definierend".
# Trotzdem: optionale MaLos (min_occurs=0) werden in Ist durchaus fehlen -> leicht erlauben.
DEFAULT_NODE_DROPOUT_BASE_BY_TYPE: Dict[str, float] = {
    "TR": 0.18,
    "NeLo": 0.06,
    "MaLo": 0.05,
    "MeLo": 0.02,
}


def _node_dropout_prob(
    node: Dict[str, Any],
    base_by_type: Dict[str, float],
    respect_min_occurs: bool = True,
) -> float:
    """
    Bestimmt Dropout-Wahrscheinlichkeit für einen konkreten Node.

    Idee:
      - Baseline pro Typ
      - Optionality/Flexibilität erhöhen Dropout
      - Mandatory Nodes (min_occurs>=1) werden stark geschützt
    """
    t = str(node.get("type"))
    p = float(base_by_type.get(t, 0.0))

    opt = _node_optionality(node)

    if respect_min_occurs and opt["min_occurs"] >= 1 and not opt["is_optional"]:
        # Mandatory -> deutlich seltener droppen
        p *= 0.15
    else:
        # Optional -> leicht erhöhen
        p *= 1.15

    if opt["is_flexible"]:
        p *= 1.10

    # Bei max_occurs>1 / unbounded ist das Objekt typischerweise variabler -> etwas mehr Dropout
    if opt["max_is_unbounded"] or (opt["max_occurs"] is not None and opt["max_occurs"] > 1):
        p *= 1.05

    # Clamp
    if p < 0.0:
        p = 0.0
    if p > 0.90:
        p = 0.90
    return p


def apply_node_dropout(
    g: TGraph,
    base_by_type: Dict[str, float] = DEFAULT_NODE_DROPOUT_BASE_BY_TYPE,
    ensure_keep_types: Tuple[str, ...] = ("MaLo", "MeLo"),
    respect_min_occurs: bool = True,
) -> Dict[str, Any]:
    """
    Entfernt zufällig Nodes aus g["nodes"] (in-place) und bereinigt betroffene Kanten.

    Wichtige Änderungen ggü. älterer Version:
      - Dropout ist optionality-aware (min/max/optional/flexibility aus node.attrs)
      - Mandatory Nodes (min_occurs>=1) werden standardmäßig geschützt

    :return: Meta-Infos: {"dropped_node_ids": [...], "dropped_counts": {...}}
    """
    nodes = list(g.get("nodes", []))
    if not nodes:
        return {}

    # Gruppiere Node-IDs nach Typ (für "ensure_keep_types").
    ids_by_type: Dict[str, List[str]] = {}
    for n in nodes:
        t = n.get("type")
        nid = n.get("id")
        if t is None or nid is None:
            continue
        ids_by_type.setdefault(str(t), []).append(str(nid))

    drop_ids: set[str] = set()
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        p = _node_dropout_prob(n, base_by_type, respect_min_occurs=respect_min_occurs)
        if p <= 0.0:
            continue
        if random.random() < p:
            drop_ids.add(str(nid))

    # Mindest-Kern-Struktur bewahren:
    for t in ensure_keep_types:
        t = str(t)
        existing = ids_by_type.get(t, [])
        if not existing:
            continue
        kept = [nid for nid in existing if nid not in drop_ids]
        if kept:
            continue
        # Alles gedroppt -> genau einen wieder behalten
        nid_keep = random.choice(existing)
        drop_ids.discard(nid_keep)

    if not drop_ids:
        return {}

    new_nodes = [n for n in nodes if str(n.get("id")) not in drop_ids]

    valid_ids = {str(n.get("id")) for n in new_nodes if n.get("id") is not None}
    edges = list(g.get("edges", []))
    new_edges = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = str(e.get("src"))
        d = str(e.get("dst"))
        if s in valid_ids and d in valid_ids:
            new_edges.append(e)

    g["nodes"] = new_nodes
    g["edges"] = new_edges
    _sync_graph_attrs_counts(g)

    dropped_counts: Dict[str, int] = {}
    for nid in drop_ids:
        found_t = None
        for t, ids in ids_by_type.items():
            if nid in ids:
                found_t = t
                break
        dropped_counts[found_t or "UNKNOWN"] = dropped_counts.get(found_t or "UNKNOWN", 0) + 1

    return {
        "dropped_node_ids": sorted(drop_ids),
        "dropped_counts": dropped_counts,
    }


def _make_unique_id(existing_ids: set[str], base: str) -> str:
    """
    Erzeugt eine neue ID, die garantiert nicht in existing_ids vorkommt.
    """
    base = base or "NODE"
    for _ in range(1000):
        suffix = random.randint(1000, 9999)
        nid = f"{base}__EXTRA_{suffix}"
        if nid not in existing_ids:
            return nid
    i = 0
    while True:
        nid = f"{base}__EXTRA_{i}"
        if nid not in existing_ids:
            return nid
        i += 1


def _candidate_melo_sources_for_dst(g: TGraph, dst_id: str, rel: str) -> List[str]:
    """
    Findet MeLo-Quellen für eine Kante mit gegebenem rel, die auf dst_id zeigt.
    Falls keine existieren, leere Liste.
    """
    rel = str(rel).upper()
    out: List[str] = []
    for e in g.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        if str(e.get("rel", "")).upper() != rel:
            continue
        if str(e.get("dst")) != str(dst_id):
            continue
        out.append(str(e.get("src")))
    return out


def apply_extra_nodes_from_optionality(
    g: TGraph,
    p_add: float = 0.25,
    max_extra_total: int = 3,
    allowed_types: Tuple[str, ...] = ("TR",),  # optional: ("TR","MaLo")
    inherit_attrs: bool = True,
    p_blank_attrs: float = 0.35,
) -> Dict[str, Any]:
    """
    Fügt zusätzliche Nodes hinzu, aber NUR dort, wo das Template es plausibel macht:
      - node.attrs.max_occurs ist unbounded ("N") oder >1
      - und (optional) node.attrs.flexibility ist "flexibel" (falls vorhanden)

    Default: nur TR, weil das in deiner Ist-Datenlage typischerweise am variabelsten ist.

    Für jeden neuen Node wird eine passende Kante von einem MeLo hinzugefügt:
      - TR  -> METR
      - MaLo-> MEMA
      - NeLo-> MENE

    :return: Meta-Infos
    """
    if max_extra_total <= 0:
        return {}

    nodes = list(g.get("nodes", []))
    if not nodes:
        return {}

    if random.random() > float(p_add):
        return {}

    existing_ids = {str(n.get("id")) for n in nodes if n.get("id") is not None}
    melo_ids = [str(n.get("id")) for n in nodes if n.get("type") == "MeLo" and n.get("id") is not None]

    # Kandidaten, die überhaupt "mehrfach" vorkommen dürfen
    candidates: List[Dict[str, Any]] = []
    for n in nodes:
        t = str(n.get("type"))
        if t not in allowed_types:
            continue
        opt = _node_optionality(n)
        if not (opt["max_is_unbounded"] or (opt["max_occurs"] is not None and opt["max_occurs"] > 1)):
            continue
        # Wenn wir flexibility kennen und sie "starr" ist: lieber nicht duplizieren
        if "flexibility" in (n.get("attrs") or {}) and not opt["is_flexible"]:
            continue
        candidates.append(n)

    if not candidates:
        return {}

    edges = list(g.get("edges", []))
    edge_set = {_edge_key(e) for e in edges if isinstance(e, dict)}

    added_ids: List[str] = []
    attached = 0

    k_total = random.randint(1, max_extra_total)
    for _ in range(k_total):
        src_node = random.choice(candidates)
        t = str(src_node.get("type"))
        src_id = str(src_node.get("id"))

        new_id = _make_unique_id(existing_ids, base=t)
        existing_ids.add(new_id)

        # attrs: entweder erben (realistische Duplikate) oder leer (Ist oft dünn)
        new_attrs: Dict[str, Any] = {}
        if inherit_attrs and isinstance(src_node.get("attrs"), dict):
            new_attrs = copy.deepcopy(src_node["attrs"])
        if new_attrs and random.random() < float(p_blank_attrs):
            # "Ist-TR haben oft wenig attrs" -> lösche die meisten, aber lass optional object_code/level stehen
            for key in list(new_attrs.keys()):
                if key in ("object_code", "level"):
                    continue
                # direction ist wichtig, aber in Ist auch oft fehlend -> 50/50 behalten
                if key == "direction" and random.random() < 0.5:
                    continue
                del new_attrs[key]

        nodes.append({"id": new_id, "type": t, "attrs": new_attrs})
        added_ids.append(new_id)

        # passende Relationen
        rel = {"TR": "METR", "MaLo": "MEMA", "NeLo": "MENE"}.get(t)
        if rel and melo_ids:
            # bevorzugt: gleiche Quellen wie das Original (falls schon verbunden), sonst random MeLo
            src_candidates = _candidate_melo_sources_for_dst(g, dst_id=src_id, rel=rel)
            src_candidates = [s for s in src_candidates if s in melo_ids] or melo_ids
            melo_src = random.choice(src_candidates)

            new_e = {"src": melo_src, "dst": new_id, "rel": rel}
            k_e = _edge_key(new_e)
            if k_e not in edge_set:
                edges.append(new_e)
                edge_set.add(k_e)
                attached += 1

    g["nodes"] = nodes
    g["edges"] = edges
    _sync_graph_attrs_counts(g)

    return {"added": len(added_ids), "added_ids": added_ids, "attached_edges": attached}


# =============================================================================
# Attachment-Mehrdeutigkeit aus Template-Regeln (graph_attrs["attachment_rules"])
# =============================================================================

def apply_attachment_ambiguity(
    g: TGraph,
    p_instantiate_from_rules: float = 0.90,
    p_rewire_existing: float = 0.35,
    only_rewire_when_rules_exist: bool = True,
) -> Tuple[int, int]:
    """
    Attachment-Mehrdeutigkeit abbilden.

    - Wenn graph_attrs["attachment_rules"] Kandidaten-MeLos nennt, instanziieren wir
      (mit Wahrscheinlichkeit) eine konkrete Anbindung (z.B. METR: MeLo -> TR).
    - Zusätzlich können wir bestehende METR/MENE/MEMA Kanten, für die Regeln existieren,
      auf alternative MeLo-Kandidaten umverdrahten.

    return: (added_edges, rewired_edges)
    """
    nodes = g.get("nodes", [])
    node_ids = {n.get("id") for n in nodes}
    melo_ids = [n.get("id") for n in nodes if n.get("type") == "MeLo"]

    if len(melo_ids) < 2:
        return 0, 0

    rules = (g.get("graph_attrs") or {}).get("attachment_rules") or []
    if not isinstance(rules, list) or not rules:
        return 0, 0

    allowed_rels = {"METR", "MENE", "MEMA"}

    rule_map: Dict[Tuple[str, str], List[str]] = {}
    for r in rules:
        if not isinstance(r, dict):
            continue
        rel = r.get("rel")
        obj = r.get("object_code")
        cands = r.get("target_candidates") or []
        if rel not in allowed_rels:
            continue
        if not obj or not isinstance(cands, list):
            continue
        valid_cands = [c for c in cands if c in node_ids]
        if not valid_cands:
            continue
        rule_map[(rel, obj)] = valid_cands

    if not rule_map:
        return 0, 0

    edges = list(g.get("edges", []))
    edge_set = {_edge_key(e) for e in edges if isinstance(e, dict)}

    added = 0
    for (rel, obj), cands in rule_map.items():
        if obj not in node_ids:
            continue
        already = any((e.get("rel") == rel and e.get("dst") == obj) for e in edges if isinstance(e, dict))
        if already:
            continue
        if random.random() > p_instantiate_from_rules:
            continue

        src = random.choice(cands)
        new_e = {"src": src, "dst": obj, "rel": rel}
        k = _edge_key(new_e)
        if k not in edge_set:
            edges.append(new_e)
            edge_set.add(k)
            added += 1

    rewired = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        rel = e.get("rel")
        dst = e.get("dst")
        src = e.get("src")
        if rel not in allowed_rels:
            continue

        key = (rel, dst)
        if key not in rule_map:
            if only_rewire_when_rules_exist:
                continue
            cands = [m for m in melo_ids if m != src]
        else:
            cands = [c for c in rule_map[key] if c != src]

        if not cands:
            continue
        if random.random() > p_rewire_existing:
            continue

        new_src = None
        for _ in range(5):
            cand = random.choice(cands)
            tentative = (str(cand), str(dst), str(rel))
            if tentative not in edge_set:
                new_src = cand
                break
        if new_src is None:
            continue

        old_key = _edge_key(e)
        edge_set.discard(old_key)
        e["src"] = new_src
        edge_set.add(_edge_key(e))
        rewired += 1

    g["edges"] = edges
    return added, rewired


# =============================================================================
# Phase A Orchestrierung
# =============================================================================

def _force_one_change(g: TGraph) -> Dict[str, Any]:
    """
    Fallback, falls durch Zufall in einem Paar keine Augmentation gegriffen hat.
    Ziel: Positive Paare sollen i.d.R. nicht 1:1 identisch bleiben.
    """
    # 1) Versuche, direction/function/dynamic o.ä. zu löschen
    candidates = []
    for node in g.get("nodes", []):
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue
        for key in ("direction", "tr_type_code", "art_der_technischen_ressource", "function", "dynamic"):
            if key in attrs:
                candidates.append((node, key))
    if candidates:
        node, key = random.choice(candidates)
        del node["attrs"][key]
        return {"forced": {"type": "attr_dropout", "node_id": node.get("id"), "attr": key}}

    # 2) Falls keine Attribute droppbar sind: droppe eine Kante
    edges = g.get("edges", [])
    if isinstance(edges, list) and edges:
        removed = edges.pop(random.randrange(len(edges)))
        g["edges"] = edges
        return {"forced": {"type": "edge_dropout", "removed": removed}}

    return {}


def augment_graph_phase_a(
    g: TGraph,
    p_edge_less: float = 0.05,
    p_apply_attachment: float = 0.70,
    ensure_nontrivial: bool = True,
) -> Dict[str, Any]:
    """
    Phase A (Ist-Noise):
      (1) Attachment-Mehrdeutigkeit (regelbasiert)
      (2) Edge-less Varianten
      (3) Edge-Dropout
      (4) Attribute-Dropout
    """
    meta: Dict[str, Any] = {}

    if random.random() < p_apply_attachment:
        added, rewired = apply_attachment_ambiguity(g)
        if added or rewired:
            meta["attachment"] = {"added": added, "rewired": rewired}

    if g.get("edges") and random.random() < p_edge_less:
        removed = apply_edge_less(g)
        meta["edge_less"] = {"removed": removed}
    else:
        dropped = apply_edge_dropout(g)
        if dropped:
            meta["edge_dropout"] = {"dropped": dropped}

    attr_dropped = apply_attribute_dropout(g)
    if attr_dropped:
        meta["attr_dropout"] = {"dropped": attr_dropped}

    if ensure_nontrivial and not meta:
        meta.update(_force_one_change(g))

    return meta


# =============================================================================
# Supervision
# =============================================================================

def build_y_from_common_node_ids(g_a: TGraph, g_b: TGraph) -> List[List[int]]:
    """
    Partial Matching Supervision:
    y = [src_indices, tgt_indices] für Nodes, die in beiden Graphen vorkommen (gleiche "id").

    - robust gegenüber unterschiedlicher Knotenzahl (Nodes können gedroppt/added sein)
    - Reihenfolge: sortiert nach src_index, damit stabil/debuggbar
    """
    nodes_a = g_a.get("nodes", [])
    nodes_b = g_b.get("nodes", [])
    if not isinstance(nodes_a, list) or not isinstance(nodes_b, list):
        return [[], []]

    id_to_a: Dict[str, int] = {}
    for i, n in enumerate(nodes_a):
        nid = n.get("id")
        if nid is not None:
            id_to_a[str(nid)] = i

    id_to_b: Dict[str, int] = {}
    for j, n in enumerate(nodes_b):
        nid = n.get("id")
        if nid is not None:
            id_to_b[str(nid)] = j

    common = [nid for nid in id_to_a.keys() if nid in id_to_b]
    common.sort(key=lambda nid: id_to_a[nid])

    src_idx = [id_to_a[nid] for nid in common]
    tgt_idx = [id_to_b[nid] for nid in common]
    return [src_idx, tgt_idx]


# =============================================================================
# Paar-Generator
# =============================================================================

def build_synthetic_pairs(
    templates: List[TGraph],
    num_pos_per_template: int = 50,
    include_negative_pairs: bool = False,
    max_neg_pairs: Optional[int] = 0,
    # Phase-A Steuerparameter
    p_edge_less: float = 0.05,
    p_apply_attachment: float = 0.70,
    ensure_nontrivial: bool = True,
    # Supervision
    supervision: str = "y",   # "y" (partial matching) oder "perm" (klassisch)
    # Phase B (nur sinnvoll bei supervision="y")
    p_node_dropout: float = 0.50,
    respect_min_occurs: bool = True,
    p_add_extra_nodes: float = 0.25,
    max_extra_nodes_total: int = 3,
    extra_node_types: Tuple[str, ...] = ("TR",),
    # Optional: Template-Meta aus Graph B entfernen
    strip_template_meta_in_b: bool = True,
) -> List[TPair]:
    """
    Positive Paare (label=1):
      - (Template, permutierte Kopie)
      - Auf die permutierte Kopie werden Phase-A Augmentations angewendet (Ist-Noise)
      - Optional (Phase B): Node-Dropout + zusätzliche Nodes (partial matching)

    Supervision:
      - supervision="perm": perm-Vektor (setzt i.d.R. gleiche Knotenzahl voraus)
      - supervision="y": y=[src_indices, tgt_indices] (auch bei unterschiedlicher Knotenzahl)

    Negative Paare:
      - Für DGMC-Training typischerweise NICHT notwendig (und kann Training sogar stören),
        deshalb default include_negative_pairs=False.
      - Wenn du sie für eine spätere Graph-Pair-Klassifikation brauchst, kannst du sie aktivieren.
    """
    if supervision not in ("perm", "y"):
        raise ValueError("supervision must be 'perm' or 'y'")

    pairs: List[TPair] = []

    # Positive Paare
    for g in templates:
        for _ in range(num_pos_per_template):
            g_perm, perm = permute_graph(g)

            aug_meta = augment_graph_phase_a(
                g_perm,
                p_edge_less=p_edge_less,
                p_apply_attachment=p_apply_attachment,
                ensure_nontrivial=ensure_nontrivial,
            )

            if supervision == "y":
                # a) Node-Dropout (optional)
                if random.random() < p_node_dropout:
                    nd_meta = apply_node_dropout(
                        g_perm,
                        base_by_type=DEFAULT_NODE_DROPOUT_BASE_BY_TYPE,
                        respect_min_occurs=respect_min_occurs,
                    )
                    if nd_meta:
                        aug_meta["node_dropout"] = nd_meta

                # b) Extra Nodes (optional, optionality-aware)
                en_meta = apply_extra_nodes_from_optionality(
                    g_perm,
                    p_add=p_add_extra_nodes,
                    max_extra_total=max_extra_nodes_total,
                    allowed_types=extra_node_types,
                )
                if en_meta:
                    aug_meta["extra_nodes"] = en_meta

            # Optional: Template-Meta entfernen, damit Graph-B stärker wie "Ist" aussieht
            if strip_template_meta_in_b:
                removed_meta = strip_template_meta_attrs(g_perm)
                if removed_meta:
                    aug_meta["strip_template_meta"] = {"removed": removed_meta}

            pair: TPair = {"graph_a": g, "graph_b": g_perm, "label": 1}

            if supervision == "perm":
                pair["perm"] = perm
            else:
                pair["y"] = build_y_from_common_node_ids(g, g_perm)

            if aug_meta:
                pair["aug"] = aug_meta

            pairs.append(pair)

    # Negative Paare (optional)
    if include_negative_pairs:
        neg_candidates: List[TPair] = []
        for i, g1 in enumerate(templates):
            for j, g2 in enumerate(templates):
                if i >= j:
                    continue
                # grobe Heuristik: unterschiedliche Labels / LBS-Codes
                if g1.get("label") == g2.get("label"):
                    continue
                neg_pair: TPair = {"graph_a": g1, "graph_b": g2, "label": 0}
                if supervision == "perm":
                    neg_pair["perm"] = None
                else:
                    neg_pair["y"] = None
                neg_candidates.append(neg_pair)

        random.shuffle(neg_candidates)
        if max_neg_pairs is not None:
            neg_candidates = neg_candidates[:max_neg_pairs]
        pairs.extend(neg_candidates)

    random.shuffle(pairs)
    return pairs


def write_pairs_jsonl(pairs: List[TPair], out_path: str) -> None:
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _print_quick_stats(pairs: List[TPair], supervision: str = "y") -> None:
    pos = [p for p in pairs if p.get("label") == 1]
    neg = [p for p in pairs if p.get("label") == 0]
    pos_with_aug = [p for p in pos if p.get("aug")]

    def count_aug(key: str) -> int:
        return sum(1 for p in pos_with_aug if key in (p.get("aug") or {}))

    print("Paare gesamt:", len(pairs), "| pos:", len(pos), "| neg:", len(neg))
    print("Positive Paare mit aug:", len(pos_with_aug), f"({len(pos_with_aug)/max(1,len(pos))*100:.1f}%)")
    print("  - attr_dropout:", count_aug("attr_dropout"))
    print("  - edge_dropout:", count_aug("edge_dropout"))
    print("  - edge_less:", count_aug("edge_less"))
    print("  - attachment:", count_aug("attachment"))
    print("  - node_dropout:", count_aug("node_dropout"))
    print("  - extra_nodes:", count_aug("extra_nodes"))
    print("  - forced:", count_aug("forced"))

    if supervision == "y":
        ratios = []
        size_diffs = 0
        for p in pos:
            ga = p.get("graph_a", {})
            gb = p.get("graph_b", {})
            na = len((ga.get("nodes") or [])) if isinstance(ga.get("nodes"), list) else 0
            nb = len((gb.get("nodes") or [])) if isinstance(gb.get("nodes"), list) else 0
            if na != nb:
                size_diffs += 1
            y = p.get("y") or [[], []]
            m = len(y[0]) if isinstance(y, list) and len(y) == 2 else 0
            ratios.append(m / max(1, na))
        if ratios:
            avg = sum(ratios) / len(ratios)
            print(f"Positive Paare mit unterschiedlicher Knotenzahl: {size_diffs} ({size_diffs/max(1,len(pos))*100:.1f}%)")
            print(f"Durchschnittliche Match-Abdeckung (|y|/|A|): {avg:.3f}")


if __name__ == "__main__":
    random.seed(42)

    base = os.path.dirname(os.path.abspath(__file__))

    # an deine aktuelle Datei angepasst
    templates_path = os.path.join(base, "data", "lbs_soll_graphs.jsonl")
    out_path = os.path.join(base, "data", "synthetic_training_pairs.jsonl")

    templates = load_templates_jsonl(templates_path)
    print("Geladene Template-Graphen:", len(templates))

    # Für reines DGMC-Training ist "perm" meist der kompatibelste Modus.
    # Für Stress-Tests / Feasibility (partial matching) kannst du auf "y" umstellen.
    supervision = "y"
    pairs = build_synthetic_pairs(
        templates,
        num_pos_per_template=60,
        include_negative_pairs=False,
        max_neg_pairs=0,
        p_edge_less=0.05,
        p_apply_attachment=0.70,
        ensure_nontrivial=True,
        supervision=supervision,
        # Phase B (nur wenn supervision="y")
        p_node_dropout=0.50,
        respect_min_occurs=True,
        p_add_extra_nodes=0.25,
        max_extra_nodes_total=3,
        extra_node_types=("TR",),
    )
    _print_quick_stats(pairs, supervision=supervision)

    write_pairs_jsonl(pairs, out_path)
    print("JSONL geschrieben nach:", out_path)
