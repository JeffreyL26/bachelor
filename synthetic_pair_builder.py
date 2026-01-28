from __future__ import annotations

"""
synthetic_pair_builder.py
Erzeugt synthetische Trainingspaare für DGMC, die nur Variationen enthalten,
graph_pipeline.py als Features sieht:
  - Nodes: Type + direction
  - Edges: rel als One-Hot

Es werden also keine Attribute wie "function", "dynamic" oder so gedroppt, diese
sind exklusiv in dne Templates

Optionality-Block (min_occurs, max_occurs,flexibility, optional) wird in
Generator-Logik genutzt (Node droppen oder Extra-Nodes)
"""

import copy
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

# JSON-Graph und Trainingspaar
TGraph = Dict[str, Any]
TPair = Dict[str, Any]


# ------------------------------
# PIPELINE-ATTRIBUTE
# ------------------------------

PIPELINE_NODE_ATTRIBUTE: Dict[str, set[str]] = {
    "MaLo": {"direction"},
    "MeLo": {"direction"},
    "TR": {"direction", "tr_type_code", "art_der_technischen_ressource"},
    "NeLo": set(),
}
PIPELINE_NODE_ATTRIBUTE_SCHLUESSEL: set[str] = set().union(*PIPELINE_NODE_ATTRIBUTE.values())


def _filter_pipeline_node_attrs(ntype: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nur besagte Attribute behalten
    """
    schluessel = PIPELINE_NODE_ATTRIBUTE.get(str(ntype), set())
    if not schluessel or not isinstance(attrs, dict):
        return {}
    return {s: attrs[s] for s in schluessel if s in attrs}


# ------------------------------
# TEMPLATES LADEN & PERMUTIEREN
# ------------------------------

def template_loader(path: str) -> List[TGraph]:
    """
    Lädt normalisierte Template-Graphen aus der JSONL

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


def graph_permutierer(g: TGraph) -> Tuple[TGraph, List[int]]:
    """
    Erzeugt eine permutierte Kopie eines Graphen:
      - Knotenreihenfolge wird zufällig vertauscht
      - Knoten-IDs bleiben gleich
      - Kanten bleiben über src/dst-IDs definiert

    :return: Permutationsliste mit perm[i] = Index des Knoten i (im Original) in der neuen Knotenliste
    :param: Besagter Graph
    """
    nodes = list(g.get("nodes") or [])
    n = len(nodes)

    reihenfolge = list(range(n))
    random.shuffle(reihenfolge)

    neue_nodes = [copy.deepcopy(nodes[i]) for i in reihenfolge]
    id_zu_neuer_id = {node.get("id"): idx for idx, node in enumerate(neue_nodes) if node.get("id") is not None}

    perm: List[int] = []
    for i in range(n):
        node_id = nodes[i].get("id")
        perm.append(id_zu_neuer_id.get(node_id, -1))

    neuer_graph: TGraph = {
        "graph_id": f'{g.get("graph_id", "")}|perm',
        "label": g.get("label"),
        "nodes": neue_nodes,
        "edges": copy.deepcopy(list(g.get("edges", []))),
        "graph_attrs": copy.deepcopy(dict(g.get("graph_attrs", {}))),
    }
    return neuer_graph, perm


# ------------------------------
# CONSTRAINT WERKZEUGE
# ------------------------------

def _als_int(wert: Any, default: int = 0) -> int:
    try:
        return int(wert)
    except Exception:
        return default


def _is_keine_obergrenze(wert: Any) -> bool:
    if wert is None:
        return False
    if isinstance(wert, str) and wert.strip().upper() == "N":
        return True
    return False


def _node_optionality(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrahiert optionale Metainfos aus node["attrs"], wenn da
    Diese Keys existieren in einem Soll-Graphen (_lbs_optionality):
      - min_occurs (int)
      - max_occurs (int bzw."N")
      - flexibility ("starr"/"flexibel")
      - optional (bool)

    Nicht relevant für DGMC, nur für Paar-Generator
    """
    attrs = node.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    min_occurs = attrs.get("min_occurs")
    max_occurs = attrs.get("max_occurs")
    optional_check = attrs.get("optional")

    is_optional = bool(optional_check) if optional_check is not None else (_als_int(min_occurs, 0) == 0)

    flexibility = attrs.get("flexibility")
    flex = str(flexibility).lower().strip() if isinstance(flexibility, str) else ""

    max_int = None
    if _is_keine_obergrenze(max_occurs):
        max_int = None
    else:
        if isinstance(max_occurs, int):
            max_int = max_occurs
        elif isinstance(max_occurs, str) and max_occurs.strip().isdigit():
            max_int = int(max_occurs.strip())
        else:
            max_int = None

    return {
        "is_optional": is_optional,
        "min_occurs": _als_int(min_occurs, 0),
        "max_occurs": max_int,          # None = unbounded/unknown
        "max_is_unbounded": _is_keine_obergrenze(max_occurs),
        "is_flexible": flex.startswith("flex"),
    }


def _edge_schluessel(edge: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(edge.get("src")), str(edge.get("dst")), str(edge.get("rel")))


def _typ_zaehler(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    anzahl = {"MaLo": 0, "MeLo": 0, "TR": 0, "NeLo": 0}
    for n in nodes:
        t = n.get("type")
        if t in anzahl:
            anzahl[t] += 1
    return anzahl


def _sync_graph_attrs_counts(g: TGraph) -> None:
    """
    Hält optionale Zählwerte in graph_attrs konsistent, wenn Knoten entfernt/ergänzt wurden
    """
    graph_attribute = g.get("graph_attrs")
    if not isinstance(graph_attribute, dict):
        return

    nodes = g.get("nodes", [])
    if not isinstance(nodes, list):
        return

    counts = _typ_zaehler(nodes)

    if "malo_count" in graph_attribute:
        graph_attribute["malo_count"] = counts["MaLo"]
    if "melo_count" in graph_attribute:
        graph_attribute["melo_count"] = counts["MeLo"]
    if "tr_count" in graph_attribute:
        graph_attribute["tr_count"] = counts["TR"]
    if "nelo_count" in graph_attribute:
        graph_attribute["nelo_count"] = counts["NeLo"]


# ------------------------------
# METADATA AUS SYN.-GRAPH RAUS
# ------------------------------

# Keys sind nur in Templates und nicht in Ist-Graphen
TEMPLATE_META_ATTR_KEYS: set[str] = {
    "min_occurs",
    "max_occurs",
    "flexibility",
    "optional",
    "object_code",
    "level",
    "reference_to_melo",
}


def template_meta_entferner(
    g: TGraph,
    keys: set[str] = TEMPLATE_META_ATTR_KEYS,
) -> int:
    """
    Entfernt Template-spezifische Meta-Attribute aus allen Nodes eines Graphen
    """
    removed = 0
    for knoten in g.get("nodes", []) or []:
        attrs = knoten.get("attrs")
        if not isinstance(attrs, dict):
            continue
        for k in list(attrs.keys()):
            if k in keys:
                del attrs[k]
                removed += 1
    return removed


# ------------------------------
# ATTRIBUTE DROP
# ------------------------------

#fehlende direction-Infos simulieren
ATTR_DROP_PROBABILITIES: Dict[str, Dict[str, float]] = {
    "MaLo": {"direction": 0.25},
    "MeLo": {"direction": 0.20},
    #fallback-keys
    "TR": {"direction": 0.25, "tr_type_code": 0.10, "art_der_technischen_ressource": 0.10},
    "NeLo": {},
}

# beeinflusst direkt edge_index/edge_attr in pipeline
# Alle auf 0, da Struktur für uns wichtigstes Signal ist
EDGE_DROP_PROPBABILITIES: Dict[str, float] = {
    "MEMA": 0.0,
    "METR": 0.0,
    "MENE": 0.0,
    "MEME": 0.0,
    "*": 0.0,  # fallback für unbekannte rels
}


def attribut_drop(
    g: TGraph,
    attribut_prob: Dict[str, Dict[str, float]] = ATTR_DROP_PROBABILITIES
) -> int:
    """
    Entfernt direction aus node["attrs"], um fehlende Information zu simulieren
    """
    dropped = 0
    for node in g.get("nodes", []):
        ntype = str(node.get("type"))
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue

        probs = attribut_prob.get(ntype, {})
        for key, p in probs.items():
            if key not in PIPELINE_NODE_ATTRIBUTE_SCHLUESSEL:
                continue
            if key in attrs and random.random() < float(p):
                del attrs[key]
                dropped += 1
    return dropped

#derzeit nicht in Benutzung
def edge_drop(
    g: TGraph,
    edge_prob: Dict[str, float] = EDGE_DROP_PROPBABILITIES
) -> int:
    """
    Entfernt zufällig Kanten aus g["edges"]
    """
    edges = list(g.get("edges", []))
    if not edges:
        return 0

    behalten: List[Dict[str, Any]] = []
    dropped = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        rel = e.get("rel") or e.get("type") or e.get("edge_type")
        rel = str(rel).strip().upper() if rel is not None else None
        p = float(edge_prob.get(rel, edge_prob.get("*", 0.0)))
        if p > 0.0 and random.random() < p:
            dropped += 1
            continue

        #normalisieren auf "rel"
        e2 = dict(e)
        if rel is not None:
            e2["rel"] = rel
        behalten.append(e2)

    g["edges"] = behalten
    return dropped

def edge_less_eigenschaft(g: TGraph) -> int:
    """
    Entfernt alle Kanten (edge-less Variante)
    """
    n = len(g.get("edges", []))
    g["edges"] = []
    return n


# ------------------------------
# PARTIELLES MATCHING
# ------------------------------

NODE_DROP_PROBABILITIES: Dict[str, float] = {
    "TR": 0.18,
    "NeLo": 0.06,
    "MaLo": 0.05,
    "MeLo": 0.02,
}


def _node_dropout_prob(
    node: Dict[str, Any],
    standard: Dict[str, float],
    min_occurs_beachten: bool = True,
) -> float:
    """
    Bestimmt Dropout-Wahrscheinlichkeit für node
    Mit Baseline pro Typ, um Pflichtkardinalitäten zu erfüllen
    """
    node_type = str(node.get("type"))
    p = float(standard.get(node_type, 0.0))

    opt = _node_optionality(node)

    if min_occurs_beachten and opt["min_occurs"] >= 1 and not opt["is_optional"]:
        p *= 0.15  # Pflicht
    else:
        p *= 1.15  # Optional

    if opt["is_flexible"]:
        p *= 1.10

    if opt["max_is_unbounded"] or (opt["max_occurs"] is not None and opt["max_occurs"] > 1):
        p *= 1.05

    return max(0.0, min(0.90, p))


def node_drop(
    g: TGraph,
    standard: Dict[str, float] = NODE_DROP_PROBABILITIES,
    typen_behalten: Tuple[str, ...] = ("MaLo", "MeLo"),
    min_occurs_beachten: bool = True,
) -> Dict[str, Any]:
    """
    Entfernt zufällig Nodes aus g["nodes"] und behandelt Kanten entsprechend

    :return: Meta-Infos für Tracking: {"dropped_node_ids": [...], "dropped_counts": {...}}
    """
    nodes = list(g.get("nodes", []))
    if not nodes:
        return {}

    ids_nach_typ: Dict[str, List[str]] = {}
    for n in nodes:
        t = n.get("type")
        nid = n.get("id")
        if t is None or nid is None:
            continue
        ids_nach_typ.setdefault(str(t), []).append(str(nid))

    drop_ids: set[str] = set()
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        p = _node_dropout_prob(n, standard, min_occurs_beachten=min_occurs_beachten)
        if p > 0.0 and random.random() < p:
            drop_ids.add(str(nid))

    # Kern bewahren (Min)
    for t in typen_behalten:
        t = str(t)
        vorhanden = ids_nach_typ.get(t, [])
        if not vorhanden:
            continue
        behalten = [nid for nid in vorhanden if nid not in drop_ids]
        if behalten:
            continue
        drop_ids.discard(random.choice(vorhanden))

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
        for t, ids in ids_nach_typ.items():
            if nid in ids:
                found_t = t
                break
        dropped_counts[found_t or "UNKNOWN"] = dropped_counts.get(found_t or "UNKNOWN", 0) + 1

    return {"dropped_node_ids": sorted(drop_ids), "dropped_counts": dropped_counts}


def _id_generator(existing_ids: set[str], base: str) -> str:
    """
    ID erstellen, die garantiert nicht in existing_ids vorkommt
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


def _melo_finder(g: TGraph, dst_id: str, rel: str) -> List[str]:
    """
    Findet MeLo-Quellen für eine Kante mit gegebenem rel, die auf dst_id zeigt
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


def node_add(
    g: TGraph,
    adding_prob: float = 0.25,
    max_extra: int = 3,
    erlaubte_nodes: Tuple[str, ...] = ("TR",),
    attribute_erben: bool = True,
    neue_nodes_unknown_dir: float = 0.05,
) -> Dict[str, Any]:
    """
    Fügt zusätzliche Nodes hinzu, aber NUR dort, wo das Template es plausibel macht:
      - max_occurs unbounded ("N") oder >1
      - und flexibility="flexibel"

    Für jeden neuen Node wird eine passende Kante von einem MeLo hinzugefügt:
      - TR  -> METR
      - MaLo-> MEMA
      - NeLo-> MENE
    """
    if max_extra <= 0:
        return {}

    nodes = list(g.get("nodes", []))
    if not nodes:
        return {}

    if random.random() > float(adding_prob):
        return {}

    vorhandene_ids = {str(n.get("id")) for n in nodes if n.get("id") is not None}
    melo_ids = [str(n.get("id")) for n in nodes if n.get("type") == "MeLo" and n.get("id") is not None]

    #Kandidaten, die überhaupt mehrfach vorkommen dürfen
    kandidaten: List[Dict[str, Any]] = []
    for n in nodes:
        t = str(n.get("type"))
        if t not in erlaubte_nodes:
            continue
        opt = _node_optionality(n)
        if not (opt["max_is_unbounded"] or (opt["max_occurs"] is not None and opt["max_occurs"] > 1)):
            continue
        if "flexibility" in (n.get("attrs") or {}) and not opt["is_flexible"]:
            continue
        kandidaten.append(n)

    if not kandidaten:
        return {}

    edges = list(g.get("edges", []))
    edge_set = {_edge_schluessel(e) for e in edges if isinstance(e, dict)}

    added_ids: List[str] = []
    attached = 0

    k_total = random.randint(1, max_extra)
    for _ in range(k_total):
        src_node = random.choice(kandidaten)
        t = str(src_node.get("type"))
        src_id = str(src_node.get("id"))

        new_id = _id_generator(vorhandene_ids, base=t)
        vorhandene_ids.add(new_id)

        #nur Pipeline-relevante Keys übernehmen
        neue_attribute: Dict[str, Any] = {}
        if attribute_erben and isinstance(src_node.get("attrs"), dict):
            neue_attribute = _filter_pipeline_node_attrs(t, copy.deepcopy(src_node["attrs"]))

        #Optional: direction unknown
        if neue_attribute and random.random() < float(neue_nodes_unknown_dir):
            #sonst könnte die Richtung bei TR über Fallbacks trotzdem gleich bleiben.
            for k in list(neue_attribute.keys()):
                if k in PIPELINE_NODE_ATTRIBUTE.get(t, set()):
                    del neue_attribute[k]

        nodes.append({"id": new_id, "type": t, "attrs": neue_attribute})
        added_ids.append(new_id)

        #passende Relationen
        rel = {"TR": "METR", "MaLo": "MEMA", "NeLo": "MENE"}.get(t)
        if rel and melo_ids:
            src_candidates = _melo_finder(g, dst_id=src_id, rel=rel)
            src_candidates = [s for s in src_candidates if s in melo_ids] or melo_ids
            melo_src = random.choice(src_candidates)

            neue_edge = {"src": melo_src, "dst": new_id, "rel": rel}
            k_e = _edge_schluessel(neue_edge)
            if k_e not in edge_set:
                edges.append(neue_edge)
                edge_set.add(k_e)
                attached += 1

    g["nodes"] = nodes
    g["edges"] = edges
    _sync_graph_attrs_counts(g)

    return {"added": len(added_ids), "added_ids": added_ids, "attached_edges": attached}


# ------------------------------
# ATTACHMENT RULES BEACHTEN
# ------------------------------

def attachment_beachten(
    g: TGraph,
    instantiate_prob: float = 0.40,
    umhaengen_existierend_prob: float = 0.2,
    nur_umhaengen_wenn_regeln_zutreffen: bool = True,
) -> Tuple[int, int]:
    """
    Kanten ergänzen oder bestehende umhängen

    :return (added_edges, rewired_edges)
    """
    nodes = g.get("nodes", [])
    node_ids = {n.get("id") for n in nodes}
    melo_ids = [n.get("id") for n in nodes if n.get("type") == "MeLo"]

    if len(melo_ids) < 2:
        return 0, 0

    rules = (g.get("graph_attrs") or {}).get("attachment_rules") or []
    if not isinstance(rules, list) or not rules:
        return 0, 0

    erlaubte_kanten = {"METR", "MENE", "MEMA"}

    regel_map: Dict[Tuple[str, str], List[str]] = {}

    #valide Kandidaten anhand Regeln
    for r in rules:
        if not isinstance(r, dict):
            continue
        rel = r.get("rel")
        rel = str(rel).strip().upper() if rel is not None else None
        obj = r.get("object_code")
        kandidaten = r.get("target_candidates") or []
        if rel not in erlaubte_kanten:
            continue
        if not obj or not isinstance(kandidaten, list):
            continue
        valide_kandidaten = [c for c in kandidaten if c in node_ids]
        if not valide_kandidaten:
            continue
        regel_map[(rel, obj)] = valide_kandidaten

    if not regel_map:
        return 0, 0

    edges = list(g.get("edges", []))
    edge_set = {_edge_schluessel(e) for e in edges if isinstance(e, dict)}

    added = 0
    for (rel, obj), kandidaten in regel_map.items():
        if obj not in node_ids:
            continue
        already = any((str(e.get("rel", "")).upper() == rel and e.get("dst") == obj) for e in edges if isinstance(e, dict))
        if already:
            continue
        if random.random() > instantiate_prob:
            continue

        src = random.choice(kandidaten)
        new_e = {"src": src, "dst": obj, "rel": rel}
        k = _edge_schluessel(new_e)
        if k not in edge_set:
            edges.append(new_e)
            edge_set.add(k)
            added += 1

    umgebaut = 0

    #durch alle edges und die mit den Regeln umbauen
    for e in edges:
        if not isinstance(e, dict):
            continue
        rel = str(e.get("rel", "")).upper()
        dst = e.get("dst")
        src = e.get("src")
        if rel not in erlaubte_kanten:
            continue

        key = (rel, dst)
        if key not in regel_map:
            if nur_umhaengen_wenn_regeln_zutreffen:
                continue
            kandidaten = [m for m in melo_ids if m != src]
        else:
            kandidaten = [c for c in regel_map[key] if c != src]

        if not kandidaten:
            continue
        if random.random() > umhaengen_existierend_prob:
            continue

        new_src = None
        for _ in range(5):
            cand = random.choice(kandidaten)
            tentative = (str(cand), str(dst), str(rel))
            if tentative not in edge_set:
                new_src = cand
                break
        if new_src is None:
            continue

        alter_key = _edge_schluessel(e)
        edge_set.discard(alter_key)
        e["src"] = new_src
        e["rel"] = rel
        edge_set.add(_edge_schluessel(e))
        umgebaut += 1

    g["edges"] = edges
    return added, umgebaut


# ------------------------------
# ÄNDERUNGEN HERVORRUFEN
# ------------------------------

def _aenderung_erzwingen(g: TGraph) -> Dict[str, Any]:
    """
    Dafür sorgen, dass in einem Paar mindestens eie Änderung ist
    """
    # Node-Attribute löschen
    kandidaten: List[Tuple[Dict[str, Any], str]] = []
    for node in g.get("nodes", []):
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue
        for key in PIPELINE_NODE_ATTRIBUTE_SCHLUESSEL:
            if key in attrs:
                kandidaten.append((node, key))

    if kandidaten:
        node, key = random.choice(kandidaten)
        del node["attrs"][key]
        return {"forced": {"type": "attr_dropout", "node_id": node.get("id"), "attr": key}}

    # Falls das nicht geklappt hat, Kante droppen
    edges = g.get("edges", [])
    if isinstance(edges, list) and edges:
        removed = edges.pop(random.randrange(len(edges)))
        g["edges"] = edges
        return {"forced": {"type": "edge_dropout", "removed": removed}}

    return {}


def graph_noisifizierer(
    g: TGraph,
    edge_less_prob: float = 0.05,
    attachment_prob: float = 0.70,
    nontrivial: bool = True,                #Wenn attachment rules keine Änderungen bewirken (weil keine Regeln), nochmal versuchen bis Änderungen oder unmöglich
) -> Dict[str, Any]:
    """
      -Kanten hinzufügen oder umhängen
      -Edge-less Varianten
      -Edge-Drop
      -direction-Attribute-Drop
    """
    meta: Dict[str, Any] = {}

    if random.random() < attachment_prob:
        added, rewired = attachment_beachten(g)
        if added or rewired:
            meta["attachment"] = {"added": added, "rewired": rewired}

    if g.get("edges") and random.random() < edge_less_prob:
        removed = edge_less_eigenschaft(g)
        meta["edge_less"] = {"removed": removed}
    else:
        dropped = edge_drop(g)
        if dropped:
            meta["edge_dropout"] = {"dropped": dropped}

    attr_dropped = attribut_drop(g)
    if attr_dropped:
        meta["attr_dropout"] = {"dropped": attr_dropped}

    if nontrivial and not meta:
        meta.update(_aenderung_erzwingen(g))

    return meta


# ------------------------------
# DGMC INPUT
# ------------------------------

def y_builder(g_a: TGraph, g_b: TGraph) -> List[List[int]]:
    """
    Partial Matching:
    y = [src_indices, tgt_indices] für Nodes, die in beiden Graphen vorkommen (gleiche "id")

    - robust gegenüber unterschiedlicher Knotenzahl (Nodes können gedroppt/added sein)
    - Reihenfolge: sortiert nach src_index, damit stabil
    """
    nodes1 = g_a.get("nodes", [])
    nodes2 = g_b.get("nodes", [])
    if not isinstance(nodes1, list) or not isinstance(nodes2, list):
        return [[], []]

    id_zu_a: Dict[str, int] = {}
    for i, n in enumerate(nodes1):
        nid = n.get("id")
        if nid is not None:
            id_zu_a[str(nid)] = i

    id_zu_b: Dict[str, int] = {}
    for j, n in enumerate(nodes2):
        nid = n.get("id")
        if nid is not None:
            id_zu_b[str(nid)] = j

    gemeinsam = [nid for nid in id_zu_a.keys() if nid in id_zu_b]
    gemeinsam.sort(key=lambda nid: id_zu_a[nid])

    src_index = [id_zu_a[nid] for nid in gemeinsam]
    tgt_index = [id_zu_b[nid] for nid in gemeinsam]
    return [src_index, tgt_index]


# ------------------------------
# PAARE GENERIEREN
# ------------------------------

def build_synthetic_pairs(
    templates: List[TGraph],
    num_pos_per_template: int = 50,
    include_negative_pairs: bool = False,
    max_neg_pairs: Optional[int] = 0,
    #Noise-Steuerung
    p_edge_less: float = 0.05,
    p_apply_attachment: float = 0.70,
    ensure_nontrivial: bool = True,
    #Partial Matching
    supervision: str = "y",   # "y" (partial matching) oder "perm" (klassisch)
    #Bei Partial Matching:
    node_drop_prob: float = 0.50,
    respect_min_occurs: bool = True,
    node_add_prob: float = 0.25,
    max_extra_nodes: int = 3,
    extra_node_types: Tuple[str, ...] = ("TR",),
    #Meta für Templates entfernen
    template_meta_raus: bool = True,
) -> List[TPair]:
    """
    Nur positive Paare (label=1) nutzen, negative bei DGMC nicht nötig:
      - (Template, permutierte Kopie)
      - Noise (edges und direction)
      - Nodes mehr/weniger

    Supervision, also Ground-Truth:
      - supervision="perm": Permutations-Vektor (gleiche Knotenzahl nötig)
      - supervision="y": y=[src_indices, tgt_indices] (partial matching)
    """
    if supervision not in ("perm", "y"):
        raise ValueError("supervision must be 'perm' or 'y'")

    pairs: List[TPair] = []

    #Positive Paare
    for graphs in templates:
        for _ in range(num_pos_per_template):
            g_perm, perm = graph_permutierer(graphs)

            aug_meta = graph_noisifizierer(
                g_perm,
                edge_less_prob=p_edge_less,
                attachment_prob=p_apply_attachment,
                nontrivial=ensure_nontrivial,
            )

            if supervision == "y":
                # Node-Dropout (optional)
                if random.random() < node_drop_prob:
                    dropped_node_meta = node_drop(
                        g_perm,
                        standard=NODE_DROP_PROBABILITIES,
                        min_occurs_beachten=respect_min_occurs,
                    )
                    if dropped_node_meta:
                        aug_meta["node_dropout"] = dropped_node_meta

                # Extra nodes (optional), entsprechend Kardinalitäten
                extra_node_meta = node_add(
                    g_perm,
                    adding_prob=node_add_prob,
                    max_extra=max_extra_nodes,
                    erlaubte_nodes=extra_node_types,
                )
                if extra_node_meta:
                    aug_meta["extra_nodes"] = extra_node_meta

            #Template-Metadaten raus
            if template_meta_raus:
                removed_meta = template_meta_entferner(g_perm)
                if removed_meta:
                    aug_meta["strip_template_meta"] = {"removed": removed_meta}

            paar: TPair = {"graph_a": graphs, "graph_b": g_perm, "label": 1}

            if supervision == "perm":
                paar["perm"] = perm
            else:
                paar["y"] = y_builder(graphs, g_perm)

            if aug_meta:
                paar["aug"] = aug_meta

            pairs.append(paar)

    #Negative Paare eig unnötig TODO
    if include_negative_pairs:
        neg_kandidaten: List[TPair] = []
        for i, g1 in enumerate(templates):
            for j, g2 in enumerate(templates):
                if i >= j:
                    continue
                if g1.get("label") == g2.get("label"):
                    continue
                neg_paare: TPair = {"graph_a": g1, "graph_b": g2, "label": 0}
                if supervision == "perm":
                    neg_paare["perm"] = None
                else:
                    neg_paare["y"] = None
                neg_kandidaten.append(neg_paare)

        random.shuffle(neg_kandidaten)
        if max_neg_pairs is not None:
            neg_kandidaten = neg_kandidaten[:max_neg_pairs]
        pairs.extend(neg_kandidaten)

    random.shuffle(pairs)
    return pairs


def paare_zu_jsonl(pairs: List[TPair], out_path: str) -> None:
    """
    Schreibt die gebauten Paare in eine JSONL-Datei
    :param pairs: Die Menge der Paare
    :param out_path: Ausgabe/Speicherpfad
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _print_stats(pairs: List[TPair], supervision: str = "y") -> None:
    """
    Konsolenausgabe über Infos zu den Paaren
    :param supervision: Welche Konfiguration man nutzt (y|perm)
    """
    pos = [p for p in pairs if p.get("label") == 1]
    neg = [p for p in pairs if p.get("label") == 0]
    pos_mit_aug = [p for p in pos if p.get("aug")]

    def augmentations_zaehler(key: str) -> int:
        return sum(1 for p in pos_mit_aug if key in (p.get("aug") or {}))

    print("Paare gesamt:", len(pairs), "| pos:", len(pos), "| neg:", len(neg))
    print("Positive Paare mit aug:", len(pos_mit_aug), f"({len(pos_mit_aug)/max(1,len(pos))*100:.1f}%)")
    print("  - attr_dropout:", augmentations_zaehler("attr_dropout"))
    print("  - edge_dropout:", augmentations_zaehler("edge_dropout"))
    print("  - edge_less:", augmentations_zaehler("edge_less"))
    print("  - attachment:", augmentations_zaehler("attachment"))
    print("  - node_dropout:", augmentations_zaehler("node_dropout"))
    print("  - extra_nodes:", augmentations_zaehler("extra_nodes"))
    print("  - forced:", augmentations_zaehler("forced"))
    print("  - strip_template_meta:", augmentations_zaehler("strip_template_meta"))

    if supervision == "y":
        ratios = []
        groessen_diff = 0
        for p in pos:
            ga = p.get("graph_a", {})
            gb = p.get("graph_b", {})
            anzahl_a = len((ga.get("nodes") or [])) if isinstance(ga.get("nodes"), list) else 0
            anzahl_b = len((gb.get("nodes") or [])) if isinstance(gb.get("nodes"), list) else 0
            if anzahl_a != anzahl_b:
                groessen_diff += 1
            y = p.get("y") or [[], []]
            m = len(y[0]) if isinstance(y, list) and len(y) == 2 else 0
            ratios.append(m / max(1, anzahl_a))
        if ratios:
            avg = sum(ratios) / len(ratios)
            print(f"Positive pairs with different node counts: {groessen_diff} ({groessen_diff/max(1,len(pos))*100:.1f}%)")
            print(f"Average of actual correspondences for graph a (|labeled correspondences in y|/|nodes in graph a|): {avg:.3f}")


if __name__ == "__main__":
    random.seed(42)

    base = os.path.dirname(os.path.abspath(__file__))

    templates_path = os.path.join(base, "data", "lbs_soll_graphs.jsonl")
    out_path = os.path.join(base, "data", "synthetic_training_pairs.jsonl")

    templates = template_loader(templates_path)
    print("Geladene Template-Graphen:", len(templates))

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
        node_drop_prob=0.50,
        respect_min_occurs=True,
        node_add_prob=0.25,
        max_extra_nodes=3,
        extra_node_types=("TR",),
        template_meta_raus=True,
    )
    _print_stats(pairs, supervision=supervision)

    paare_zu_jsonl(pairs, out_path)
    print("JSONL written to:", out_path)
