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

    Wichtig: Die permutierte Kopie ist eine deepcoyp der Knoten/Kanten/Attrs,
    damit spätere Veränderungen nicht das Original-Template mutieren.

    :param g: Graph
    :return: (Permutierte Kopie, Permutationsliste)
    """
    nodes = list(g["nodes"])
    n = len(nodes)

    order = list(range(n))
    random.shuffle(order)

    # wir wollen später Attribute/Kanten "kaputt" machen, ohne das Original zu ändern (deepcopy)
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


def apply_attribute_dropout(g: TGraph, probs_by_type: Dict[str, Dict[str, float]] = DEFAULT_ATTR_DROPOUT_BY_TYPE) -> int:
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


def apply_edge_dropout(g: TGraph, drop_by_rel: Dict[str, float] = DEFAULT_EDGE_DROPOUT_BY_REL) -> int:
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
      - Zusätzlich können wir bestehende METR/MENE Kanten, für die Regeln existieren,
        auf alternative MeLo-Kandidaten umverdrahten.

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

    # Map (rel, object_code) -> candidates
    rule_map: Dict[Tuple[str, str], List[str]] = {}
    for r in rules:
        if not isinstance(r, dict):
            continue
        rel = r.get("rel")
        obj = r.get("object_code")
        cands = r.get("target_candidates") or []
        if rel not in ("METR", "MENE"):
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
    # 1) Regeln instanziieren: fehlende METR/MENE Kanten hinzufügen
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
        if rel not in ("METR", "MENE"):
            continue

        key = (rel, dst)
        if key not in rule_map:
            if only_rewire_when_rules_exist:
                continue
            # fallback: alle MeLo als Kandidaten
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


def augment_graph_phase_a(
    g: TGraph,
    p_edge_less: float = 0.05,
    p_apply_attachment: float = 0.70,
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

    return meta


# ------------------------------
# SYNTHETISCHE PAARE BAUEN
# ------------------------------

def build_synthetic_pairs(
    templates: List[TGraph],
    num_pos_per_template: int = 20,
    max_neg_pairs: Optional[int] = 200,
    # Phase-A Steuerparameter
    p_edge_less: float = 0.05,
    p_apply_attachment: float = 0.70,
) -> List[TPair]:
    """
    Baut synthetische Trainingspaare:

    Positive Paare (label=1):
      - (Template, permutierte Kopie)
      - Auf die permutierte Kopie werden Phase-A Augmentations angewendet:
          (1) Attribute-Dropout
          (2) Edge-Dropout
          (3) Edge-less Varianten
          (5) Attachment-Mehrdeutigkeit (regelbasiert)

    Negative Paare (label=0):
      - (Template_i, Template_j) mit i != j und unterschiedlichem Label/LBS-Code

    :param templates: Liste von Template-Graphen (Soll)
    :param num_pos_per_template: Anzahl der positiven Paare pro Template
    :param max_neg_pairs: Max. Anzahl negativer Paare insgesamt
    :param p_edge_less: Wahrscheinlichkeit, alle Kanten zu entfernen (wenn Kanten existieren)
    :param p_apply_attachment: Wahrscheinlichkeit, Attachment-Ambiguity-Operator anzuwenden
    :return: Liste von Paaren
    """
    pairs: List[TPair] = []

    # Positive Paare
    for g in templates:
        for _ in range(num_pos_per_template):
            g_perm, perm = permute_graph(g)

            aug_meta = augment_graph_phase_a(
                g_perm,
                p_edge_less=p_edge_less,
                p_apply_attachment=p_apply_attachment,
            )

            pair: TPair = {
                "graph_a": g,
                "graph_b": g_perm,
                "label": 1,
                "perm": perm,
            }
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
            neg_candidates.append({
                "graph_a": g1,
                "graph_b": g2,
                "label": 0,
                "perm": None,
            })

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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Reproduzierbarkeit
    random.seed(42)

    base = os.path.dirname(os.path.abspath(__file__))

    templates_path = os.path.join(base, "data", "lbs_soll_graphs_pro.jsonl")
    out_path       = os.path.join(base, "data", "synthetic_training_pairs.jsonl")

    templates = load_templates_jsonl(templates_path)
    print("Geladene Template-Graphen:", len(templates))

    pairs = build_synthetic_pairs(
        templates,
        num_pos_per_template=20,
        max_neg_pairs=200,
        p_edge_less=0.05,
        p_apply_attachment=0.70,
    )
    print("Gebildete Paare:", len(pairs))

    write_pairs_jsonl(pairs, out_path)
    print("JSONL geschrieben nach:", out_path)
