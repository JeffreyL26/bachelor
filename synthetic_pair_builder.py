from __future__ import annotations
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

#TODO: Kommentare und Dokumentation revamp - Ziel 15.12

# JSON-Graph und Trainingspaar
TGraph = Dict[str, Any]
TPair  = Dict[str, Any]

# ------------------------------
# GRAPHEN LADEN UND PERMUTIEREN
# ------------------------------

def load_templates_jsonl(path: str) -> List[TGraph]:
    """
    Lädt alle normalisierte Template-Graphen aus einer JSONL-Datei,
    aus @graph_templates.build_all_templates
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
    - die Knoten-IDs bleiben gleich
    - Kanten bleiben als (src/dst)-IDs erhalten
    → G2 hat dieselbe Struktur wie G1, Reihenfolge der Knoten anders, wahre Permutation der Knoten hinterlegen
    Zusätzlich wird eine Permutationsliste zurückgegeben:
    perm[i] = Index des Knoten i (im Original) in der neuen Knotenliste.

    Ziel ist es, dass DGMC weiß, dass ein Knoten dem Knoten in einem anderen Graphen entspricht, ganz gleich wie sie nummeriert sind
    :param g: Graph
    :return: Permutierte Kopie, Permutationsliste
    """
    # Knotenliste kopieren
    nodes = list(g["nodes"])
    n = len(nodes)

    # Zufallsreihenfolge
    order = list(range(n))
    random.shuffle(order)

    # Neue Knoten
    new_nodes = [nodes[i] for i in order]

    # Node-ID → neuer Index
    id_to_new_idx = {node["id"]: idx for idx, node in enumerate(new_nodes)}

    # Dient zur Ground-Truth-Zuordnung (Matching-Matrix) die DGMC lernt
    # perm[i] = j, Knoten, der früher an Stelle i war, ist jetzt an Stelle j
    perm = [id_to_new_idx[nodes[i]["id"]] for i in range(n)]

    # Permutierten Graphen bauen
    new_g: TGraph = {
        "graph_id": f'{g.get("graph_id", "")}|perm',                    # perm anhängen → permutierte Version des Original-Graphen
        "label": g.get("label"),
        "nodes": new_nodes,                                             # "Neue" Knoten
        "edges": list(g.get("edges", [])),                              # 1:1 dieselben, sie speichern die ID und brauchen daher keine Indizes
        "graph_attrs": dict(g.get("graph_attrs", {})),
    }
    return new_g, perm


# ------------------------------
# SYNTHETISCHE PAARE BAUEN
# ------------------------------

def build_synthetic_pairs(
    templates: List[TGraph],
    num_pos_per_template: int = 20,
    max_neg_pairs: Optional[int] = 200,
) -> List[TPair]:
    """
    Baut synthetische Trainingspaare:
    - Positive Paare: (Template, permutierte Kopie des gleichen Templates) mit label=1 und Permutationsliste
    - Negative Paare: (Template_i, Template_j) mit i != j und unterschiedlichem LBS-Code, label=0, perm=None
    """

    pairs: List[TPair] = []

    # 1) Positive Paare (Permutation desselben Graphen)
    for g in templates:
        for _ in range(num_pos_per_template):
            g_perm, perm = permute_graph(g)
            pairs.append({
                "graph_a": g,
                "graph_b": g_perm,
                "label": 1,
                "perm": perm,  # Ground Truth: index graph_a -> index graph_b
            })

    # 2) Negative Paare (verschiedene LBS-Codes)
    # Wir nehmen alle Kombinationen und schneiden ggf. nach max_neg_pairs ab.
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

    # 3) Gesamte Paare mischen
    random.shuffle(pairs)
    return pairs


def write_pairs_jsonl(pairs: List[TPair], out_path: str) -> None:
    """
    Schreibt die Paare als JSONL-Datei, eine Zeile pro Paar.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    random.seed(42)

    base = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(base, "data", "lbs_soll_graphs.jsonl")
    out_path       = os.path.join(base, "data", "synthetic_training_pairs.jsonl")

    templates = load_templates_jsonl(templates_path)
    print("Geladene Template-Graphen:", len(templates))

    pairs = build_synthetic_pairs(
        templates,
        num_pos_per_template=20,
        max_neg_pairs=200,
    )
    print("Gebildete Paare:", len(pairs))

    write_pairs_jsonl(pairs, out_path)
    print("JSONL geschrieben nach:", out_path)