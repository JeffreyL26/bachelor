from __future__ import annotations

"""
synthetic_pair_builder_control.py
Permutation-Vergleichsgruppe für DGMC

Wenn man so will "Permutation-Only-Baseline":
- Graph B ist eine permutierte Kopie von Graph A (gleiche Knotenmenge)
- y = [[src_local...],[tgt_local...]] (vollständiges Matching, nicht partial wie in @synthetic_pair_builder.py)
"""

import os
import random
from typing import Any, Dict, List

from synthetic_pair_builder import (
    _print_stats,
    attachment_beachten,
    attribut_drop,
    edge_drop,
    edge_less_eigenschaft,
    y_builder,
    template_loader,
    graph_permutierer,
    paare_zu_jsonl,
)

TGraph = Dict[str, Any]
TPair = Dict[str, Any]


# ------------------------------
# ATTRIBUTE DROPPEN
# ------------------------------

#Optionaler Noise, aber diesmal auf 1:1-Korrespondenz (derzeit aus)
ATTR_DROP_PROBABILITIES: Dict[str, Dict[str, float]] = {
    "MaLo": {"direction": 0.25},
    "MeLo": {"direction": 0.20},
    "TR": {"direction": 0.25, "tr_type_code": 0.10, "art_der_technischen_ressource": 0.10},
    "NeLo": {},
}

#eh aus
EDGE_DROP_PROBABILITIES: Dict[str, float] = {
    "MEMA": 0.10,
    "METR": 0.30,
    "MENE": 0.20,
    "MEME": 0.10,
    "*": 0.15,
}

#1. Tabelle von oben mit Faktor skalieren
def _attr_drop_skalieren(prob_tabelle: Dict[str, Dict[str, float]], faktor: float) -> Dict[str, Dict[str, float]]:
    """
    Erste Tabelle @ATTR_DROP_PROBABILITIES mit Wahrscheinlichkeitsfaktor skalieren
    """
    faktor = float(faktor)
    if faktor <= 0.0:
        return {keys: {inner_key: 0.0 for inner_key in inner_dict.keys()} for keys, inner_dict in prob_tabelle.items()}
    return {keys: {inner_key: float(drop_prob) * faktor for inner_key, drop_prob in inner_dict.items()} for keys, inner_dict in prob_tabelle.items()}


#2. Tabelle von oben mit Faktor skalieren
def _edge_drop_skalieren(prob_tabelle: Dict[str, float], faktor: float) -> Dict[str, float]:
    """
    Wie @_attr_drop_skalieren, nur für die 2. Tabelle
    """
    faktor = float(faktor)
    if faktor <= 0.0:
        return {keys: 0.0 for keys in prob_tabelle.keys()}
    return {keys: float(drop_prob) * faktor for keys, drop_prob in prob_tabelle.items()}


def perm_paare_bauen(
    templates_path: str,
    out_path: str,
    *,
    seed: int = 42,
    num_pos_per_template: int = 60,
    # Optionaler Noise
    # Skalierungsfaktoren: 0.0 => aus, 1.0 => dann greift oben
    edge_drop_faktor: float = 0.0,
    attr_drop_faktor: float = 0.0,
    edge_less_prob: float = 0.0,
    # Attachment-Rules instanziieren/umhängen (falls es graph_attrs["attachment_rules"] gibt)
    attachment_umhaengen_ins: float = 0.0,
    # Wenn True und keine Noise-Regel gegriffen hat: erzwinge MINDESTENS eine Feature-relevante Änderung.
    # Für eine Kontrollgruppe ist das i.d.R. False.
    ensure_nontrivial: bool = False,
    # Optional: zusätzlich perm mitschreiben (Debugging/Analyse). DGMC ignoriert es.
    include_perm_field: bool = False,
) -> None:
    """
    Baut synthetische Trainingspaare mit 1:1-Korrespondenz und schreibt sie in eine JSONL
    """

    random.seed(seed)

    templates = template_loader(templates_path)
    print("Loaded Template-Graphs:", len(templates))

    # Pre-scale dropout tables once.
    edge_drop_table = _edge_drop_skalieren(EDGE_DROP_PROBABILITIES, edge_drop_faktor)
    attr_drop_table = _attr_drop_skalieren(ATTR_DROP_PROBABILITIES, attr_drop_faktor)

    pairs: List[TPair] = []

    for graph in templates:
        for _ in range(num_pos_per_template):
            g_perm, perm = graph_permutierer(graph)
            aug_meta: Dict[str, Any] = {}

            # Optionale Attachments
            if float(attachment_umhaengen_ins) > 0.0 and random.random() < float(attachment_umhaengen_ins):
                added, rewired = attachment_beachten(g_perm)
                if added or rewired:
                    aug_meta["attachment"] = {"added": added, "rewired": rewired}

            # Optional: Edge-less
            edge_less_passiert = False
            if float(edge_less_prob) > 0.0 and g_perm.get("edges") and random.random() < float(edge_less_prob):
                entfernte_edges = edge_less_eigenschaft(g_perm)
                aug_meta["edge_less"] = {"removed": entfernte_edges}
                edge_less_passiert = True

            # Optional: Edge-Drop
            if (not edge_less_passiert) and edge_drop_faktor > 0.0:
                dropped = edge_drop(g_perm, edge_prob=edge_drop_table)
                if dropped:
                    aug_meta["edge_dropout"] = {"dropped": dropped}

            # Optional: Attribute-Drop
            if attr_drop_faktor > 0.0:
                dropped_attribute = attribut_drop(g_perm, attribut_prob=attr_drop_table)
                if dropped_attribute:
                    aug_meta["attr_dropout"] = {"dropped": dropped_attribute}

            # Änderung erzwingen, wenn Noise aktiv
            if ensure_nontrivial and not aug_meta:
                aenderung_erzwungen = False
                if attr_drop_faktor > 0.0:
                    dropped_attribute = attribut_drop(
                        g_perm,
                        attribut_prob=_attr_drop_skalieren(ATTR_DROP_PROBABILITIES, 1.0),
                    )
                    if dropped_attribute:
                        aug_meta["forced"] = {"type": "attr_dropout", "dropped": dropped_attribute}
                        aenderung_erzwungen = True
                if (not aenderung_erzwungen) and edge_drop_faktor > 0.0 and g_perm.get("edges"):
                    dropped = edge_drop(
                        g_perm,
                        edge_prob=_edge_drop_skalieren(EDGE_DROP_PROBABILITIES, 1.0),
                    )
                    if dropped:
                        aug_meta["forced"] = {"type": "edge_dropout", "dropped": dropped}
                        aenderung_erzwungen = True

            # DGMC 1:1-Zuordnung
            y = y_builder(graph, g_perm)

            pair: TPair = {"graph_a": graph, "graph_b": g_perm, "label": 1, "y": y}
            if include_perm_field:
                pair["perm"] = perm
            if aug_meta:
                pair["aug"] = aug_meta

            pairs.append(pair)

    random.shuffle(pairs)

    _print_stats(pairs, supervision="y")
    paare_zu_jsonl(pairs, out_path)
    print("JSONL written to:", out_path)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    templates_path = os.path.join(base, "data", "lbs_soll_graphs.jsonl")
    out_path = os.path.join(base, "data", "synthetic_training_pairs_control.jsonl")

    #Default
    perm_paare_bauen(
        templates_path,
        out_path,
        seed=42,
        num_pos_per_template=60,
        edge_drop_faktor=0.0,
        attr_drop_faktor=0.0,
        edge_less_prob=0.0,
        attachment_umhaengen_ins=0.0,
        ensure_nontrivial=False,
        include_perm_field=False,
    )
