from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch

from graph_pipeline import json_graph_to_pyg
from dgmc import DGMC
from dgmc.models import GIN


JsonGraph = Dict[str, Any]


# -----------------------------
# Hilfsfunktionen
# -----------------------------

def load_jsonl_graphs(path: str) -> List[JsonGraph]:
    graphs: List[JsonGraph] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            graphs.append(json.loads(line))
    return graphs


def build_model(in_channels: int,
                hidden_dim: int = 64,
                num_steps: int = 10,
                device: torch.device = torch.device("cpu")) -> DGMC:
    """
    Baut die DGMC-Architektur genau so nach, wie du sie beim Training verwendet hast.
    """
    psi_1 = GIN(in_channels, hidden_dim, num_layers=3)
    psi_2 = GIN(in_channels, hidden_dim, num_layers=3)
    model = DGMC(psi_1, psi_2, num_steps=num_steps, k=-1).to(device)
    return model


@torch.no_grad()
def match_pair(
    model: DGMC,
    g_src: JsonGraph,
    g_tgt: JsonGraph,
    device: torch.device,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Berechnet einen Matching-Score zwischen einem Ist-Graphen (g_src)
    und einem Template-Graphen (g_tgt) sowie ein Knoten-Mapping.

    Rückgabe:
    - score: globaler Score (Mean der row-wise Maxima der Matching-Matrix)
    - mapping: Liste von Zuordnungen pro Quellknoten mit Score
    """

    # JSON -> PyG-Graphen
    data_s = json_graph_to_pyg(g_src, undirected=True)
    data_t = json_graph_to_pyg(g_tgt, undirected=True)

    data_s = data_s.to(device)
    data_t = data_t.to(device)

    # DGMC-Forward, analog zum Training (ohne y)
    S0, SL = model(
        data_s.x, data_s.edge_index, data_s.edge_attr, None,
        data_t.x, data_t.edge_index, data_t.edge_attr, None,
    )

    # Wir nehmen die finale Matching-Matrix SL
    S = SL  # Shape [num_src_nodes, num_tgt_nodes]

    # Zeilenweise Softmax → pro Quellknoten eine Verteilung über Zielknoten
    S_soft = torch.softmax(S, dim=1)

    # Row-wise Maxima als Score pro Quellknoten
    row_max_vals, row_max_idx = S_soft.max(dim=1)  # Shape [num_src_nodes]

    # Globaler Graph-Score: Mittelwert der row-wise Maxima
    score = row_max_vals.mean().item()

    # Node-IDs zur besseren Interpretierbarkeit
    src_ids = [n["id"] for n in g_src.get("nodes", [])]
    tgt_ids = [n["id"] for n in g_tgt.get("nodes", [])]

    mapping: List[Dict[str, Any]] = []
    num_src = len(src_ids)
    for i in range(num_src):
        j = int(row_max_idx[i].item())
        src_node_id = src_ids[i] if i < len(src_ids) else None
        tgt_node_id = tgt_ids[j] if j < len(tgt_ids) else None
        mapping.append({
            "src_index": i,
            "src_node_id": src_node_id,
            "tgt_index": j,
            "tgt_node_id": tgt_node_id,
            "score": float(row_max_vals[i].item()),
        })

    return score, mapping


# -----------------------------
# Hauptpipeline: Ist -> Template
# -----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = os.path.dirname(os.path.abspath(__file__))

    # Pfade an dein Repo angepasst:
    ist_path = os.path.join(base, "ist_graphs.jsonl")
    if not os.path.exists(ist_path):
        # Fallback: vielleicht hast du die Datei in data/ gelegt
        ist_path = os.path.join(base, "data", "ist_graphs.jsonl")

    templates_path = os.path.join(base, "data", "lbs_soll_graphs.jsonl")
    model_path = os.path.join(base, "data", "dgmc_templates.pt")
    out_path = os.path.join(base, "data", "ist_dgmc_matches.jsonl")

    print("Lade Ist-Graphen aus:", ist_path)
    ist_graphs = load_jsonl_graphs(ist_path)
    print("Anzahl Ist-Graphen:", len(ist_graphs))

    print("Lade Template-Graphen aus:", templates_path)
    template_graphs = load_jsonl_graphs(templates_path)
    print("Anzahl Templates:", len(template_graphs))

    # Templates vorbereiten: PyG-Feature-Dimension bestimmen und Pattern merken
    if not template_graphs:
        raise RuntimeError("Keine Templates gefunden – bitte graph_templates.py ausführen.")

    # Feature-Dimension aus erstem Template bestimmen
    from graph_pipeline import json_graph_to_pyg as _j2p
    sample_template_pyg = _j2p(template_graphs[0], undirected=True)
    in_channels = sample_template_pyg.x.size(-1)

    model = build_model(in_channels, hidden_dim=64, num_steps=10, device=device)
    if not os.path.exists(model_path):
        raise RuntimeError(f"DGMC-Modell nicht gefunden unter {model_path}. Bitte zuerst dgmc_template_training.py ausführen.")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("DGMC-Modell geladen von:", model_path)

    # Templates nach Pattern gruppieren (MaLo:MeLo-Verhältnis)
    templates_by_pattern: Dict[str, List[JsonGraph]] = {}
    for g in template_graphs:
        pat = g.get("graph_attrs", {}).get("pattern", "unknown")
        templates_by_pattern.setdefault(pat, []).append(g)

    print("Template-Patterns:", {k: len(v) for k, v in templates_by_pattern.items()})

    # Ist-Graphen matchen
    num_ist = len(ist_graphs)
    written = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, g_ist in enumerate(ist_graphs):
            g_attrs = g_ist.get("graph_attrs", {})
            ist_pattern = g_attrs.get("pattern", "unknown")

            # Kandidaten-Templates: gleiche Pattern, sonst fallback auf alle
            candidates = templates_by_pattern.get(ist_pattern)
            if not candidates:
                candidates = template_graphs

            best_score = float("-inf")
            best_template: JsonGraph | None = None
            best_mapping: List[Dict[str, Any]] = []

            for g_tpl in candidates:
                score, mapping = match_pair(model, g_ist, g_tpl, device)
                if score > best_score:
                    best_score = score
                    best_template = g_tpl
                    best_mapping = mapping

            result = {
                "ist_graph_id": g_ist.get("graph_id"),
                "ist_pattern": ist_pattern,
                "ist_malo_count": g_attrs.get("malo_count"),
                "ist_melo_count": g_attrs.get("melo_count"),
                "best_template_graph_id": best_template.get("graph_id") if best_template else None,
                "best_template_label": best_template.get("label") if best_template else None,
                "best_template_pattern": best_template.get("graph_attrs", {}).get("pattern") if best_template else None,
                "best_score": best_score,
                "mapping": best_mapping,
            }

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

            if (idx + 1) % 100 == 0:
                print(f"Verarbeitet: {idx+1}/{num_ist} Ist-Graphen")

    print(f"Fertig. Ergebnisse geschrieben nach: {out_path}")
    print(f"Anzahl gematchter Ist-Graphen: {written}")


if __name__ == "__main__":
    main()
