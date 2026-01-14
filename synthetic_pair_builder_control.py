from __future__ import annotations

"""synthetic_pair_builder_control.py

Kontrollgruppe (Annäherung A) für DGMC-Experimente.

Ziel
----
Eine *saubere* Baseline, die möglichst nah an "Permutation-only" ist:

- Graph B ist eine permutierte Kopie von Graph A (gleiche Knotenmenge)
- Supervision im DGMC-Format: y = [[src_local...],[tgt_local...]] (vollständiges Matching)

Optional kann (sehr gezielt) "Ist-Noise" zugeschaltet werden – aber **nur** auf
Feature-Inputs, die die Pipeline tatsächlich in `graph_pipeline.py` encodiert.

Warum y statt perm?
-------------------
DGMC (Library/Paper) trainiert mit *sparse correspondences* y. Ein Permutationsvektor
ist nur eine alternative Darstellung derselben Zuordnung und wird von deiner
aktuellen `dgmc_dataset.py` nicht konsumiert. Daher schreibt dieses Skript y.

Hinweis
-------
Dieses Skript importiert Helper aus `synthetic_pair_builder.py` (Annäherung B),
nutzt aber **nicht** dessen Pair-Orchestrierung, damit die Kontrollgruppe wirklich
"clean" bleibt.
"""

import os
import random
from typing import Any, Dict, List

# Re-use Helper aus Annäherung B.
from synthetic_pair_builder import (
    _print_quick_stats,
    apply_attachment_ambiguity,
    apply_attribute_dropout,
    apply_edge_dropout,
    apply_edge_less,
    build_y_from_common_node_ids,
    load_templates_jsonl,
    permute_graph,
    write_pairs_jsonl,
)

TGraph = Dict[str, Any]
TPair = Dict[str, Any]


# -----------------------------------------------------------------------------
# IMPORTANT: Control should only touch attributes that the *graph_pipeline* uses.
#
# Aktueller Pipeline-Stand (graph_pipeline.py):
#   - Node: type one-hot + direction one-hot
#   - TR fallback für direction: tr_type_code / art_der_technischen_ressource
#
# Deshalb droppen wir hier ausschließlich diese Keys.
# -----------------------------------------------------------------------------

DEFAULT_ATTR_DROPOUT_BY_TYPE_CONTROL: Dict[str, Dict[str, float]] = {
    "MaLo": {"direction": 0.25},
    "MeLo": {"direction": 0.20},
    "TR": {"direction": 0.25, "tr_type_code": 0.10, "art_der_technischen_ressource": 0.10},
    "NeLo": {},
}

DEFAULT_EDGE_DROPOUT_BY_REL_CONTROL: Dict[str, float] = {
    "MEMA": 0.10,
    "METR": 0.30,
    "MENE": 0.20,
    "MEME": 0.10,
    "*": 0.15,
}


def _scale_nested_probs(d: Dict[str, Dict[str, float]], factor: float) -> Dict[str, Dict[str, float]]:
    factor = float(factor)
    if factor <= 0.0:
        return {k: {kk: 0.0 for kk in vv.keys()} for k, vv in d.items()}
    return {k: {kk: float(v) * factor for kk, v in vv.items()} for k, vv in d.items()}


def _scale_flat_probs(d: Dict[str, float], factor: float) -> Dict[str, float]:
    factor = float(factor)
    if factor <= 0.0:
        return {k: 0.0 for k in d.keys()}
    return {k: float(v) * factor for k, v in d.items()}


def build_control_pairs_A(
    templates_path: str,
    out_path: str,
    *,
    seed: int = 42,
    num_pos_per_template: int = 60,
    # -------------------------
    # Optionaler Ist-Noise (nur Feature-relevant!)
    # -------------------------
    # Skalierungsfaktoren: 0.0 => aus, 1.0 => Control-Defaults oben.
    edge_dropout_factor: float = 0.0,
    attr_dropout_factor: float = 0.0,
    # Edge-less Variante (entfernt alle Kanten) – nur sinnvoll, wenn du so einen Ist-Fall erwartest.
    p_edge_less: float = 0.0,
    # Attachment-Rules instanziieren/rewiren (falls graph_attrs["attachment_rules"] existiert)
    p_apply_attachment: float = 0.0,
    # Wenn True und keine Noise-Regel gegriffen hat: erzwinge MINDESTENS eine Feature-relevante Änderung.
    # Für eine Kontrollgruppe ist das i.d.R. False.
    ensure_nontrivial: bool = False,
    # Optional: zusätzlich perm mitschreiben (Debugging/Analyse). DGMC ignoriert es.
    include_perm_field: bool = False,
) -> None:
    """Erzeugt Kontroll-Paare und schreibt JSONL."""

    random.seed(seed)

    templates = load_templates_jsonl(templates_path)
    print("Geladene Template-Graphen:", len(templates))

    # Pre-scale dropout tables once.
    edge_drop_table = _scale_flat_probs(DEFAULT_EDGE_DROPOUT_BY_REL_CONTROL, edge_dropout_factor)
    attr_drop_table = _scale_nested_probs(DEFAULT_ATTR_DROPOUT_BY_TYPE_CONTROL, attr_dropout_factor)

    pairs: List[TPair] = []

    for g in templates:
        for _ in range(num_pos_per_template):
            g_perm, perm = permute_graph(g)
            aug_meta: Dict[str, Any] = {}

            # (1) Optional: Attachment-Ambiguity
            if float(p_apply_attachment) > 0.0 and random.random() < float(p_apply_attachment):
                added, rewired = apply_attachment_ambiguity(g_perm)
                if added or rewired:
                    aug_meta["attachment"] = {"added": added, "rewired": rewired}

            # (2) Optional: Edge-less
            did_edge_less = False
            if float(p_edge_less) > 0.0 and g_perm.get("edges") and random.random() < float(p_edge_less):
                removed_edges = apply_edge_less(g_perm)
                aug_meta["edge_less"] = {"removed": removed_edges}
                did_edge_less = True

            # (3) Optional: Edge-dropout (nur wenn nicht edge-less)
            if (not did_edge_less) and edge_dropout_factor > 0.0:
                dropped = apply_edge_dropout(g_perm, drop_by_rel=edge_drop_table)
                if dropped:
                    aug_meta["edge_dropout"] = {"dropped": dropped}

            # (4) Optional: Attr-dropout (nur pipeline-relevante keys)
            if attr_dropout_factor > 0.0:
                dropped_attrs = apply_attribute_dropout(g_perm, probs_by_type=attr_drop_table)
                if dropped_attrs:
                    aug_meta["attr_dropout"] = {"dropped": dropped_attrs}

            # (5) Optional: erzwinge eine Änderung (aber nur, wenn Noise prinzipiell aktiv ist)
            if ensure_nontrivial and not aug_meta:
                forced_done = False
                if attr_dropout_factor > 0.0:
                    dropped_attrs = apply_attribute_dropout(
                        g_perm,
                        probs_by_type=_scale_nested_probs(DEFAULT_ATTR_DROPOUT_BY_TYPE_CONTROL, 1.0),
                    )
                    if dropped_attrs:
                        aug_meta["forced"] = {"type": "attr_dropout", "dropped": dropped_attrs}
                        forced_done = True
                if (not forced_done) and edge_dropout_factor > 0.0 and g_perm.get("edges"):
                    dropped = apply_edge_dropout(
                        g_perm,
                        drop_by_rel=_scale_flat_probs(DEFAULT_EDGE_DROPOUT_BY_REL_CONTROL, 1.0),
                    )
                    if dropped:
                        aug_meta["forced"] = {"type": "edge_dropout", "dropped": dropped}
                        forced_done = True

            # DGMC-Supervision: vollständige 1:1 Zuordnung (weil keine Node-Varianz)
            y = build_y_from_common_node_ids(g, g_perm)

            pair: TPair = {"graph_a": g, "graph_b": g_perm, "label": 1, "y": y}
            if include_perm_field:
                pair["perm"] = perm
            if aug_meta:
                pair["aug"] = aug_meta

            pairs.append(pair)

    random.shuffle(pairs)

    _print_quick_stats(pairs, supervision="y")
    write_pairs_jsonl(pairs, out_path)
    print("JSONL geschrieben nach:", out_path)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    templates_path = os.path.join(base, "data", "lbs_soll_graphs.jsonl")
    out_path = os.path.join(base, "data", "synthetic_training_pairs_control.jsonl")

    # Default: echte Kontrollgruppe (Permutation-only), aber mit y (DGMC-kompatibel)
    build_control_pairs_A(
        templates_path,
        out_path,
        seed=42,
        num_pos_per_template=60,
        edge_dropout_factor=0.0,
        attr_dropout_factor=0.0,
        p_edge_less=0.0,
        p_apply_attachment=0.0,
        ensure_nontrivial=False,
        include_perm_field=False,
    )
