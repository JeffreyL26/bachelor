from __future__ import annotations
import copy
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


# JSON-Graph und Trainingspaar
TGraph = Dict[str, Any]
TPair  = Dict[str, Any]

# ------------------------------
# GRAPHEN LADEN UND PERMUTIEREN
# ------------------------------

def load_templates_jsonl(path: str) -> List[TGraph]:
    """
    Lädt alle normalisierte Template-Graphen aus einer JSONL-Datei.

    Erwartetes Format pro Zeile:
      {"graph_id": ..., "label": ..., "nodes": [...], "edges": [...], "graph_attrs": {...}}

    :param path: Pfad zur JSONL-Datei
    :return: Liste von Graphen
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

    Wichtig: Die permutierte Kopie ist eine DEEP-COPY der Knoten/Kanten/Attrs,
    damit spätere Augmentierungen nicht das Original-Template mutieren.

    :param g: Graph
    :return: (Permutierte Kopie, Permutationsliste)
    """
    nodes = list(g["nodes"])
    n = len(nodes)

    order = list(range(n))
    random.shuffle(order)

    # Deep-Copy: wir wollen später Attribute/Kanten "kaputt" machen, ohne das Original zu ändern
    new_nodes = [copy.deepcopy(nodes[i]) for i in order]
    id_to_new_idx = {node["id"]: idx for idx, node in enumerate(new_nodes)}

    perm = [id_to_new_idx[nodes[i]["id"]] for i in range(n)]

    new_g: TGraph = {
        "graph_id": f'{g.get("graph_id", "")}|perm',
        "label": g.get("label"),
        "nodes": new_nodes,
        "edges": copy.deepcopy(list(g.get("edges", []))),
        "graph_attrs": copy.deepcopy(dict(g.get("graph_attrs", {}))),
    }
    return new_g, perm


# ------------------------------
# AUGMENTATIONS (Phase A)
# ------------------------------

# Attribute-Dropout: wir simulieren "Ist-Noise" (fehlende Attribute).
# Dropout ist bewusst typ-abhängig, um fachlich plausible Lücken zu erzeugen.
DEFAULT_ATTR_DROPOUT_BY_TYPE: Dict[str, Dict[str, float]] = {
    "MaLo": {"direction": 0.30},
    "MeLo": {"function": 0.25, "voltage_level": 0.20, "dynamic": 0.20},
    "TR":   {"tr_direction": 0.30, "direction": 0.20},
    "NeLo": {},
}

# Edge-Dropout: METR häufiger droppen (TR-Beziehungen sind in Ist häufiger unvollständig)
DEFAULT_EDGE_DROPOUT_BY_REL: Dict[str, float] = {
    "MEMA": 0.10,
    "METR": 0.30,
    "MENE": 0.20,
    "MEME": 0.10,
    "*":    0.15,  # fallback für unbekannte rels
}

# ------------------------------
# PARTIAL MATCHING / Option 1 (Phase B)
# ------------------------------

# Node-Dropout: v.a. TR/NeLo sind in Ist häufiger "fehlend" oder variabel.
DEFAULT_NODE_DROPOUT_BY_TYPE: Dict[str, float] = {
    "TR":   0.20,
    "NeLo": 0.05,
    # MaLo/MeLo standardmäßig nicht droppen, weil sie i.d.R. strukturell definierend sind.
    "MaLo": 0.00,
    "MeLo": 0.00,
}


def apply_attribute_dropout(
    g: TGraph,
    probs_by_type: Dict[str, Dict[str, float]] = DEFAULT_ATTR_DROPOUT_BY_TYPE
) -> int:
    """
    Entfernt (löscht) ausgewählte Attribute aus node["attrs"], um fehlende Information zu simulieren.
    :return: Anzahl gelöschter Attribute
    """
    dropped = 0
    for node in g.get("nodes", []):
        ntype = node.get("type")
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue
        probs = probs_by_type.get(ntype, {})
        for key, p in probs.items():
            if key in attrs and random.random() < p:
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
        rel = e.get("rel")
        p = drop_by_rel.get(rel, drop_by_rel.get("*", 0.0))
        if p > 0.0 and random.random() < p:
            dropped += 1
            continue
        kept.append(e)

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
    graph_pipeline nutzt die Keys typischerweise nicht für x, aber es hilft Debugging und Konsistenz.
    """
    ga = g.get("graph_attrs")
    if not isinstance(ga, dict):
        return

    nodes = g.get("nodes", [])
    if not isinstance(nodes, list):
        return

    counts = _recount_types(nodes)

    # Beide Namensvarianten tolerieren (in manchen Dateien gab es *_count, in anderen *_min/_max).
    if "malo_count" in ga:
        ga["malo_count"] = counts["MaLo"]
    if "melo_count" in ga:
        ga["melo_count"] = counts["MeLo"]
    if "tr_count" in ga:
        ga["tr_count"] = counts["TR"]
    if "nelo_count" in ga:
        ga["nelo_count"] = counts["NeLo"]


def apply_node_dropout(
    g: TGraph,
    drop_by_type: Dict[str, float] = DEFAULT_NODE_DROPOUT_BY_TYPE,
    ensure_keep_types: Tuple[str, ...] = ("MaLo", "MeLo"),
) -> Dict[str, Any]:
    """
    Entfernt zufällig Nodes aus g["nodes"] (in-place) und bereinigt betroffene Kanten.
    Standardmäßig werden MaLo/MeLo nicht gedroppt (s.o.), und zusätzlich wird erzwungen,
    dass mindestens ein Node je Typ in ensure_keep_types übrig bleibt (falls vorhanden).

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

    # Kandidaten zum Droppen (per Node-Entscheidung).
    drop_ids: set[str] = set()
    for n in nodes:
        t = str(n.get("type"))
        nid = n.get("id")
        if nid is None:
            continue
        p = float(drop_by_type.get(t, 0.0))
        if p <= 0.0:
            continue
        if random.random() < p:
            drop_ids.add(str(nid))

    # Mindest-Kern-Struktur bewahren:
    # Wenn wir aus Versehen alles eines Kern-Typs gedroppt hätten, rollback für einen.
    for t in ensure_keep_types:
        t = str(t)
        existing = ids_by_type.get(t, [])
        if not existing:
            continue  # Typ kommt nicht vor -> egal
        kept = [nid for nid in existing if nid not in drop_ids]
        if kept:
            continue
        # Alles gedroppt -> genau einen wieder behalten
        nid_keep = random.choice(existing)
        if nid_keep in drop_ids:
            drop_ids.remove(nid_keep)

    if not drop_ids:
        return {}

    # Nodes filtern
    new_nodes = [n for n in nodes if str(n.get("id")) not in drop_ids]

    # Kanten bereinigen: nur Kanten behalten, deren src/dst noch existieren
    valid_ids = {str(n.get("id")) for n in new_nodes if n.get("id") is not None}
    edges = list(g.get("edges", []))
    new_edges = []
    for e in edges:
        s = str(e.get("src"))
        d = str(e.get("dst"))
        if s in valid_ids and d in valid_ids:
            new_edges.append(e)

    g["nodes"] = new_nodes
    g["edges"] = new_edges
    _sync_graph_attrs_counts(g)

    dropped_counts: Dict[str, int] = {}
    for nid in drop_ids:
        # Typ für Statistik herausfinden (aus ids_by_type)
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
    # extrem unwahrscheinlich
    i = 0
    while True:
        nid = f"{base}__EXTRA_{i}"
        if nid not in existing_ids:
            return nid
        i += 1


def apply_extra_tr_nodes(
    g: TGraph,
    p_add: float = 0.20,
    max_extra: int = 2,
    attach_rel: str = "METR",
) -> Dict[str, Any]:
    """
    Fügt zusätzliche TR-Nodes hinzu (in-place), um variable Kardinalitäten zu simulieren.
    Optional wird je neuem TR eine Kante (attach_rel) von einem vorhandenen MeLo zugefügt.

    :return: Meta-Infos: {"added": k, "added_ids": [...], "attached_edges": k2}
    """
    if max_extra <= 0:
        return {}

    nodes = list(g.get("nodes", []))
    if not nodes:
        return {}

    if random.random() > p_add:
        return {}

    existing_ids = {str(n.get("id")) for n in nodes if n.get("id") is not None}
    melo_ids = [str(n.get("id")) for n in nodes if n.get("type") == "MeLo" and n.get("id") is not None]

    # Anzahl neuer TR-Nodes: 1..max_extra
    k = random.randint(1, max_extra)

    added_ids: List[str] = []
    attached = 0
    edges = list(g.get("edges", []))
    edge_set = {_edge_key(e) for e in edges}

    for _ in range(k):
        new_id = _make_unique_id(existing_ids, base="TR")
        existing_ids.add(new_id)

        # Minimal plausible TR attrs: viele Ist-TR haben kaum Attribute
        nodes.append({"id": new_id, "type": "TR", "attrs": {}})
        added_ids.append(new_id)

        # Wenn möglich, an ein MeLo anbinden (METR)
        if melo_ids:
            src = random.choice(melo_ids)
            new_e = {"src": src, "dst": new_id, "rel": attach_rel}
            k_e = _edge_key(new_e)
            if k_e not in edge_set:
                edges.append(new_e)
                edge_set.add(k_e)
                attached += 1

    g["nodes"] = nodes
    g["edges"] = edges
    _sync_graph_attrs_counts(g)

    return {"added": len(added_ids), "added_ids": added_ids, "attached_edges": attached}


def apply_attachment_ambiguity(
    g: TGraph,
    p_instantiate_from_rules: float = 0.90,
    p_rewire_existing: float = 0.35,
    only_rewire_when_rules_exist: bool = True,
) -> Tuple[int, int]:
    """
    Phase A Punkt (5): Attachment-Mehrdeutigkeit abbilden.

    Idee:
      - Wenn graph_attrs["attachment_rules"] Kandidaten-MeLos nennt, instanziieren wir
        (mit Wahrscheinlichkeit) eine konkrete Anbindung (z.B. METR: MeLo -> TR).
      - Zusätzlich können wir bestehende METR/MENE/MEMA Kanten, für die Regeln existieren,
        auf alternative MeLo-Kandidaten umverdrahten.

    Das ist fachlich plausibel, weil die Kandidaten explizit aus den Templates stammen
    (keine geratenen Kanten ohne Regelbasis).

    :return: (added_edges, rewired_edges)
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

    # Map (rel, object_code) -> candidates
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
        # nur Kandidaten, die als Nodes existieren
        valid_cands = [c for c in cands if c in node_ids]
        if not valid_cands:
            continue
        rule_map[(rel, obj)] = valid_cands

    if not rule_map:
        return 0, 0

    edges = list(g.get("edges", []))
    edge_set = {_edge_key(e) for e in edges}

    added = 0
    # 1) Regeln instanziieren: fehlende METR/MENE/MEMA Kanten hinzufügen
    for (rel, obj), cands in rule_map.items():
        if obj not in node_ids:
            continue
        # existiert bereits eine Kante dieses Typs zu diesem Objekt?
        already = any((e.get("rel") == rel and e.get("dst") == obj) for e in edges)
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

    # 2) Bestehende Kanten umverdrahten, aber nur dort, wo Regeln Alternativen erlauben
    rewired = 0
    for e in edges:
        rel = e.get("rel")
        dst = e.get("dst")
        src = e.get("src")
        if rel not in allowed_rels:
            continue

        key = (rel, dst)
        if key not in rule_map:
            if only_rewire_when_rules_exist:
                continue
            # fallback (wenn man es explizit will): alle MeLo als Kandidaten
            cands = [m for m in melo_ids if m != src]
        else:
            cands = [c for c in rule_map[key] if c != src]

        if not cands:
            continue
        if random.random() > p_rewire_existing:
            continue

        # versuche ein paarmal eine nicht-duplizierte Kante zu erzeugen
        new_src = None
        for _ in range(5):
            cand = random.choice(cands)
            tentative = (str(cand), str(dst), str(rel))
            if tentative not in edge_set:
                new_src = cand
                break
        if new_src is None:
            continue

        # edge_set aktualisieren: alte Kante raus, neue rein
        old_key = _edge_key(e)
        if old_key in edge_set:
            edge_set.remove(old_key)
        e["src"] = new_src
        edge_set.add(_edge_key(e))
        rewired += 1

    g["edges"] = edges
    return added, rewired


def _force_one_change(g: TGraph) -> Dict[str, Any]:
    """
    Fallback, falls durch Zufall in einem Paar keine Augmentation gegriffen hat.
    Ziel: Positive Paare sollen i.d.R. nicht 1:1 identisch bleiben (außer wirklich nichts droppbar ist).
    """
    # 1) Versuche, ein relevantes Attribut zu löschen
    candidates = []
    for node in g.get("nodes", []):
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue
        for key in ("direction", "tr_direction", "function", "voltage_level", "dynamic"):
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
    Wendet die Phase-A Augmentations auf einen Graphen an (in-place).
    Implementiert:
      (1) Attribute-Dropout
      (2) Edge-Dropout
      (3) Edge-less Varianten
      (5) Attachment-Mehrdeutigkeit (regelbasiert)

    :return: kleines Meta-Dict (für Debug/Analyse), wird optional im Paar gespeichert
    """
    meta: Dict[str, Any] = {}

    # (5) Attachment-Varianten zuerst (kann Kanten hinzufügen/umverdrahten)
    if random.random() < p_apply_attachment:
        added, rewired = apply_attachment_ambiguity(g)
        if added or rewired:
            meta["attachment"] = {"added": added, "rewired": rewired}

    # (3) Edge-less mit kleiner Wahrscheinlichkeit (falls Kanten existieren)
    if g.get("edges") and random.random() < p_edge_less:
        removed = apply_edge_less(g)
        meta["edge_less"] = {"removed": removed}
    else:
        # (2) Edge-Dropout
        dropped = apply_edge_dropout(g)
        if dropped:
            meta["edge_dropout"] = {"dropped": dropped}

    # (1) Attribute-Dropout
    attr_dropped = apply_attribute_dropout(g)
    if attr_dropped:
        meta["attr_dropout"] = {"dropped": attr_dropped}

    # Safety: im Zweifel ein kleines bisschen Noise erzwingen,
    # damit der Datensatz nicht aus reinem "Permutation-only" besteht.
    if ensure_nontrivial and not meta:
        meta.update(_force_one_change(g))

    return meta


def build_y_from_common_node_ids(g_a: TGraph, g_b: TGraph) -> List[List[int]]:
    """
    Option 1: Supervision als Korrespondenzliste (partial matching).
    Liefert y = [src_indices, tgt_indices] für Nodes, die in beiden Graphen vorkommen (gleiche "id").

    - robust gegenüber unterschiedlicher Knotenzahl (Nodes können gedroppt/added sein)
    - Reihenfolge: sortiert nach src_index, damit stabil/debuggbar

    :return: y als zwei gleich lange Listen (src_indices, tgt_indices)
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


# ------------------------------
# SYNTHETISCHE PAARE BAUEN
# ------------------------------

def build_synthetic_pairs(
    templates: List[TGraph],
    num_pos_per_template: int = 50,
    max_neg_pairs: Optional[int] = 500,
    # Phase-A Steuerparameter
    p_edge_less: float = 0.05,
    p_apply_attachment: float = 0.70,
    ensure_nontrivial: bool = True,
    # Option 1: partial matching Supervision
    supervision: str = "y",  # "y" (korrespondenzliste) oder "perm" (alte Voll-Permutation)
    # Option 1: zusätzliche Knoten-Varianz (nur sinnvoll bei supervision="y")
    p_node_dropout: float = 0.50,
    node_dropout_by_type: Dict[str, float] = DEFAULT_NODE_DROPOUT_BY_TYPE,
    p_add_extra_tr: float = 0.25,
    max_extra_tr: int = 2,
) -> List[TPair]:
    """
    Baut synthetische Trainingspaare.

    Positive Paare (label=1):
      - (Template, permutierte Kopie)
      - Auf die permutierte Kopie werden Phase-A Augmentations angewendet:
          (1) Attribute-Dropout
          (2) Edge-Dropout
          (3) Edge-less Varianten
          (5) Attachment-Mehrdeutigkeit (regelbasiert)
      - Optional (Option 1): Knoten-Varianz durch Node-Dropout + Extra-TR-Nodes

      Supervision:
        - supervision="perm": perm-Vektor (nur bei gleicher Knotenzahl sinnvoll)
        - supervision="y": y=[src_indices, tgt_indices] (auch bei unterschiedlicher Knotenzahl)

    Negative Paare (label=0):
      - (Template_i, Template_j) mit i != j und unterschiedlichem Label/LBS-Code

    :return: Liste von Paaren
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

            # Option 1: Knoten-Varianz (nur wenn wir y als Supervision ausgeben)
            if supervision == "y":
                # a) Node-Dropout (mit Wahrscheinlichkeit) – v.a. TR/NeLo
                if random.random() < p_node_dropout:
                    nd_meta = apply_node_dropout(g_perm, drop_by_type=node_dropout_by_type)
                    if nd_meta:
                        aug_meta["node_dropout"] = nd_meta

                # b) Extra TR Nodes hinzufügen (mit Wahrscheinlichkeit)
                en_meta = apply_extra_tr_nodes(g_perm, p_add=p_add_extra_tr, max_extra=max_extra_tr)
                if en_meta:
                    aug_meta["extra_tr"] = en_meta

            pair: TPair = {
                "graph_a": g,
                "graph_b": g_perm,
                "label": 1,
            }

            # Nur noch für Testzwecke
            if supervision == "perm":
                # Hinweis: perm ist nur korrekt, solange die Knotenzahl unverändert blieb.
                pair["perm"] = perm
            else:
                pair["y"] = build_y_from_common_node_ids(g, g_perm)

            if aug_meta:
                pair["aug"] = aug_meta

            pairs.append(pair)

    # Negative Paare (unterschiedliche Labels/LBS-Codes)
    neg_candidates: List[TPair] = []
    for i, g1 in enumerate(templates):
        for j, g2 in enumerate(templates):
            if i >= j:
                continue
            if g1.get("label") == g2.get("label"):
                continue
            neg_pair: TPair = {
                "graph_a": g1,
                "graph_b": g2,
                "label": 0,
            }
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
    """
    Schreibt die Paare als JSONL-Datei (eine Zeile pro Paar).
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _print_quick_stats(pairs: List[TPair], supervision: str = "y") -> None:
    """
    Kleine Konsole-Statistik, damit sofort sichtbar ist, ob Augmentations greifen.
    """
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
    print("  - extra_tr:", count_aug("extra_tr"))
    print("  - forced:", count_aug("forced"))

    if supervision == "y":
        # Match-Ratio (wie viel der Template-Nodes tatsächlich gematcht werden)
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
            denom = max(1, na)
            ratios.append(m / denom)
        if ratios:
            avg = sum(ratios) / len(ratios)
            print(f"Positive Paare mit unterschiedlicher Knotenzahl: {size_diffs} ({size_diffs/max(1,len(pos))*100:.1f}%)")
            print(f"Durchschnittliche Match-Abdeckung (|y|/|A|): {avg:.3f}")


if __name__ == "__main__":
    # Reproduzierbarkeit
    random.seed(42)

    base = os.path.dirname(os.path.abspath(__file__))

    # Aktueller Default: pro-Templates
    templates_path = os.path.join(base, "data", "lbs_soll_graphs_pro.jsonl")
    out_path       = os.path.join(base, "data", "synthetic_training_pairs50.jsonl")

    templates = load_templates_jsonl(templates_path)
    print("Geladene Template-Graphen:", len(templates))

    supervision = "y"

    pairs = build_synthetic_pairs(
        templates,
        num_pos_per_template=50,
        max_neg_pairs=200,
        p_edge_less=0.05,
        p_apply_attachment=0.70,
        ensure_nontrivial=True,
        supervision=supervision,
        # Option 1 Defaults (anpassbar)
        p_node_dropout=0.50,
        p_add_extra_tr=0.25,
        max_extra_tr=2,
    )
    _print_quick_stats(pairs, supervision=supervision)

    write_pairs_jsonl(pairs, out_path)
    print("JSONL geschrieben nach:", out_path)
