from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple
import torch
from graph_pipeline import json_graph_to_pyg
from dgmc import DGMC
from wrapper_GINE import EdgeAwareGINE


JsonGraph = Dict[str, Any]


# ------------------------------
# TODO: Kommentare
# ------------------------------

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
                edge_dim: int,
                hidden_dim: int = 64,
                num_steps: int = 10,
                device: torch.device = torch.device("cpu")) -> DGMC:
    """
    DGMC-Architektur, wie beim Training verwendet
    """
    psi_1 = EdgeAwareGINE(in_channels, hidden_dim, edge_dim, num_layers=3, cat=True, lin=True)
    psi_2 = EdgeAwareGINE(in_channels, hidden_dim, edge_dim, num_layers=3, cat=True, lin=True)
    model = DGMC(psi_1, psi_2, num_steps=num_steps, k=-1).to(device)
    return model

def _greedy_typisiert(
    S: torch.Tensor,               # [ns, nt], bereits "wahrscheinlichkeitsartig"
    src_types: List[str],
    tgt_types: List[str],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    1:1 (partial) Matching per greedx:
    - nur Typ-kompatible Paare erlaubt (MaLo-MaLo, MeLo-MeLo, TR-TR, NeLo-NeLo)
    - jeder Target-Knoten max. 1x
    - jeder Source-Knoten max. 1x
    - Score bestraft Unmatched über Normalisierung mit max(ns, nt)
    """
    ns, nt = S.size(0), S.size(1)

    # Typ -> Indizes (effizienter als ns*nt Schleife)
    tgt_by_type: Dict[str, List[int]] = {}
    for j, t in enumerate(tgt_types):
        tgt_by_type.setdefault(t, []).append(j)

    # Kandidatenliste: (score, i, j)
    cands: List[Tuple[float, int, int]] = []
    S_cpu = S.detach().cpu()
    for i, st in enumerate(src_types):
        js = tgt_by_type.get(st, [])
        for j in js:
            sc = float(S_cpu[i, j].item())
            cands.append((sc, i, j))

    # beste Paare zuerst
    cands.sort(reverse=True, key=lambda x: x[0])

    used_src = set()
    used_tgt = set()
    chosen: Dict[int, Tuple[int, float]] = {}  # src_i -> (tgt_j, score)

    for sc, i, j in cands:
        if i in used_src or j in used_tgt:
            continue
        used_src.add(i)
        used_tgt.add(j)
        chosen[i] = (j, sc)

        if len(used_src) >= min(ns, nt):
            break

    # Mapping-Liste (für alle src-Knoten, unmatched -> tgt=None)
    mapping: List[Dict[str, Any]] = []
    sum_scores = 0.0
    for i in range(ns):
        if i in chosen:
            j, sc = chosen[i]
            sum_scores += sc
            mapping.append({
                "src_index": i,
                "tgt_index": j,
                "score": float(sc),
            })
        else:
            mapping.append({
                "src_index": i,
                "tgt_index": None,
                "score": 0.0,
            })

    # Global Score: Sum / max(ns, nt) -> penalisiert unmatched
    denom = float(max(ns, nt)) if max(ns, nt) > 0 else 1.0
    score = sum_scores / denom
    return score, mapping


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

    :return
    - score: globaler Score (Mean der row-wise Maxima der Matching-Matrix)
    - mapping: Liste von Zuordnungen pro Quellknoten mit Score
    """

    # JSON -> PyG-Graphen
    data_s = json_graph_to_pyg(g_src, undirected=True).to(device)
    data_t = json_graph_to_pyg(g_tgt, undirected=True).to(device)

    # DGMC-Forward, analog zum Training (ohne y)
    S0, SL = model(
        data_s.x, data_s.edge_index, data_s.edge_attr, None,
        data_t.x, data_t.edge_index, data_t.edge_attr, None,
    )

    # Wir nehmen die finale Matching-Matrix SL
    S = SL if model.num_steps > 0 else S0   # Shape [num_src_nodes, num_tgt_nodes]
    if hasattr(S, "to_dense"):
        S = S.to_dense()

    # Falls irgendwann sparse (k>=1) verwendet wird:
    #if getattr(S, "is_sparse", False):
    #    S = S.to_dense()

    # Row-wise Maxima als Score pro Quellknoten
    row_max_vals, row_max_idx = S.max(dim=1)  # Shape [num_src_nodes]

    # Globaler Graph-Score: Mittelwert der row-wise Maxima
    score = row_max_vals.mean().item()

    # Node-IDs zur besseren Interpretierbarkeit
    src_nodes = g_src.get("nodes", [])
    tgt_nodes = g_tgt.get("nodes", [])

    src_ids = [n.get("id") for n in src_nodes]
    tgt_ids = [n.get("id") for n in tgt_nodes]
    src_types = [n.get("type") for n in src_nodes]
    tgt_types = [n.get("type") for n in tgt_nodes]

    score, mapping_greedy = _greedy_typisiert(S, src_types, tgt_types)

    mapping: List[Dict[str, Any]] = []
    for gmp in mapping_greedy:
        i = gmp["src_index"]
        j = gmp["tgt_index"]
        mapping.append({
            "src_index": i,
            "src_node_id": src_ids[i] if i < len(src_ids) else None,
            "src_type": src_types[i] if i < len(src_types) else None,
            "tgt_index": j,
            "tgt_node_id": (tgt_ids[j] if (j is not None and j < len(tgt_ids)) else None),
            "tgt_type": (tgt_types[j] if (j is not None and j < len(tgt_types)) else None),
            "score": float(gmp["score"]),
        })

    return score, mapping


# ------------------------------
# TODO: Kommentare
# ------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = os.path.dirname(os.path.abspath(__file__))

    ist_path = os.path.join(base, "ist_graphs.jsonl")
    if not os.path.exists(ist_path):
        # Fallback
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
    edge_dim = sample_template_pyg.edge_attr.size(-1)

    model = build_model(in_channels, edge_dim=edge_dim, hidden_dim=64, num_steps=10, device=device)
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
