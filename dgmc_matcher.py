"""
dgmc_ist_matcher.py

Zweck (Thesis/Proposal-konform):
- Wendet ein trainiertes DGMC-Modell "out-of-the-box" auf reale Ist-Graphen an.
- Pro Ist-Graph wird ein bestes LBS-Template über DGMC-Ähnlichkeit ausgewählt.
- Zusätzlich werden pro Template-Knoten die Top-k Kandidaten im Ist-Graph ausgegeben
  (k = "Top-k Matches" für die praktische Nutzbarkeit / Machbarkeitsanalyse).

Wichtige Einordnung:
- DGMC ist ein *node-level* 1:1 Matching-Verfahren. Es löst keine 1:N-Entsprechungen.
- In eurem Use-Case (Ist hat ggf. mehr/weniger Knoten als Template) ist das erwartbar problematisch.
  Diese Datei versucht das nicht "zurechtzuklopfen", sondern macht das Verhalten sichtbar:
  -> Top-k Kandidaten pro Template-Knoten, ohne zusätzliche Discrepancy-Reports.

Inputs:
- Ist-Graphen:    JSONL, z.B. data/ist_graphs_all.jsonl
- Soll-Templates: JSONL, z.B. data/lbs_soll_graphs_pro.jsonl
- Modell:         runs/dgmc_y/model.pt (oder dein eigener Pfad)

Output:
- JSONL-Datei mit einem Ergebnis pro Ist-Graph:
  {
    "ist_graph_id": ...,
    "best_template_graph_id": ...,
    "best_template_label": ...,
    "best_score": float,
    "topk": [
       {
         "src_index": i,
         "src_node_id": "...",
         "src_type": "MaLo|MeLo|TR|NeLo",
         "candidates": [
            {"tgt_index": j, "tgt_node_id": "...", "tgt_type": "...", "score": ...},
            ...
         ]
       },
       ...
    ]
  }

Ausführen:
- Parameter im CONFIG-Block anpassen.
- Dann: python dgmc_ist_matcher.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

# --- DGMC robust import (je nach Installation) ---
try:
    from dgmc import DGMC  # type: ignore
except Exception:
    try:
        from dgmc.models import DGMC  # type: ignore
    except Exception:
        try:
            from torch_geometric.nn.models import DGMC  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Konnte DGMC nicht importieren. Bitte deep-graph-matching-consensus installieren "
                "(https://github.com/rusty1s/deep-graph-matching-consensus)."
            ) from e

from wrapper_GINE import EdgeAwareGINE
from graph_pipeline import json_graph_to_pyg


# ============================================================
# CONFIG (hier editieren; keine CLI-Args notwendig)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Daten
IST_PATH = BASE_DIR / "data" / "ist_graphs_all.jsonl"
TEMPLATES_PATH = BASE_DIR / "data" / "lbs_soll_graphs_pro.jsonl"

# Modell (aus dgmc_template_training.py)
MODEL_PATH = BASE_DIR / "runs" / "dgmc_y" / "model.pt"

# Output
OUT_DIR = BASE_DIR / "runs" / "dgmc_y"
OUT_JSONL = OUT_DIR / "ist_template_matches_topk.jsonl"

# Inference-Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNDIRECTED = True

# (1) Top-k Templates (Ranking) -------------------------------
# Wie viele Template-Kandidaten pro Ist-Graph exportiert werden.
K_TOP_TEMPLATES = 3

# (2) Top-k Matches (Kandidaten pro Template-Knoten) ----------
# Anzahl der Kandidaten pro Source-Knoten (Template-Knoten), die exportiert werden.
K_TOP_MATCHES = 3

# (DGMC-interne Parameter; sollten zu Training passen)
NUM_STEPS = 10
DGMC_INTERNAL_K = -1    # -1 = dense (empfohlen für kleine Graphen). Muss zum Train-Run passen.
DETACH = False

# GNN-Architektur (muss zum Train-Run passen)
HIDDEN_DIM = 64

# psi_2 (Random Indicator Dimension; muss zum Train-Run passen)
PSI2_IN = 16
PSI2_OUT = 16
PSI1_LAYERS = 3
PSI2_LAYERS = 2

# Optional: Debug Limits
MAX_IST_GRAPHS: Optional[int] = None        # z.B. 50
MAX_TEMPLATES: Optional[int] = None         # z.B. 15


# ============================================================
# Helpers: IO
# ============================================================

JsonGraph = Dict[str, Any]


def _resolve_default(path: Path) -> Path:
    """
    Falls du deine JSONL außerhalb von /data liegen hast, wird ein Fallback versucht.
    """
    if path.exists():
        return path
    alt = BASE_DIR / path.name
    if alt.exists():
        return alt
    return path  # will error later with clear message


def iter_jsonl(path: Path, max_lines: Optional[int] = None) -> Iterable[JsonGraph]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_lines is not None and i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: Path, max_lines: Optional[int] = None) -> List[JsonGraph]:
    return list(iter_jsonl(path, max_lines=max_lines))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Model construction (muss zum Training passen)
# ============================================================

class ResettableEdgeAwareGINE(torch.nn.Module):
    """
    DGMC ruft reset_parameters() auf psi_1/psi_2 in DGMC.reset_parameters().
    EdgeAwareGINE hat keine reset_parameters-Methode, daher ein Wrapper.

    WICHTIG:
    - Die Gewichte im state_dict hängen am Attributnamen `base`.
      Daher sollte dieser Wrapper auch für Inference identisch sein, damit load_state_dict passt.
    """
    def __init__(self, base: EdgeAwareGINE) -> None:
        super().__init__()
        self.base = base
        # DGMC erwartet (insb. für psi_2) in_channels/out_channels Attribute:
        self.in_channels = getattr(base, "in_channels", None)
        self.out_channels = getattr(base, "out_channels", None)

    def reset_parameters(self) -> None:  # pragma: no cover (inference)
        for m in self.modules():
            if m is self:
                continue
            rp = getattr(m, "reset_parameters", None)
            if callable(rp):
                rp()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.base(x, edge_index, edge_attr, batch=batch)


def build_model(in_dim: int, edge_dim: int, device: torch.device) -> DGMC:
    psi_1 = ResettableEdgeAwareGINE(
        EdgeAwareGINE(
            in_channels=in_dim,
            out_channels=HIDDEN_DIM,
            edge_dim=edge_dim,
            num_layers=PSI1_LAYERS,
            dropout=0.0,
            batch_norm=True,
            cat=False,
            lin=True,
        )
    )
    psi_2 = ResettableEdgeAwareGINE(
        EdgeAwareGINE(
            in_channels=PSI2_IN,
            out_channels=PSI2_OUT,
            edge_dim=edge_dim,
            num_layers=PSI2_LAYERS,
            dropout=0.0,
            batch_norm=True,
            cat=False,
            lin=True,
        )
    )
    model = DGMC(psi_1, psi_2, num_steps=NUM_STEPS, k=DGMC_INTERNAL_K, detach=DETACH).to(device)
    return model


# ============================================================
# Matching logic
# ============================================================

@dataclass
class PreparedGraph:
    graph_json: JsonGraph
    data: Any  # PyG Data
    node_ids: List[Any]
    node_types: List[Any]


def prepare_graph(g: JsonGraph, device: torch.device) -> PreparedGraph:
    """
    JSON → PyG Data.
    Wir behalten node_ids/types aus graph_pipeline (DGMC ignoriert das, aber für Output wichtig).
    """
    d = json_graph_to_pyg(g, undirected=UNDIRECTED)
    # ensure tensors on device
    d = d.to(device)
    node_ids = list(getattr(d, "node_ids", []))
    node_types = list(getattr(d, "node_types", []))
    return PreparedGraph(graph_json=g, data=d, node_ids=node_ids, node_types=node_types)


@torch.no_grad()
def dgmc_similarity_matrix(
    model: DGMC,
    src: PreparedGraph,
    tgt: PreparedGraph,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes dense similarity matrix S [ns, nt] for one (src,tgt) pair.

    We explicitly pass batch vectors (all zeros) to be robust with DGMC's internal to_dense_batch usage.
    """
    x_s = src.data.x
    ei_s = src.data.edge_index
    ea_s = getattr(src.data, "edge_attr", None)

    x_t = tgt.data.x
    ei_t = tgt.data.edge_index
    ea_t = getattr(tgt.data, "edge_attr", None)

    ns = int(x_s.size(0))
    nt = int(x_t.size(0))

    batch_s = torch.zeros(ns, dtype=torch.long, device=device)
    batch_t = torch.zeros(nt, dtype=torch.long, device=device)

    S0, SL = model(x_s, ei_s, ea_s, batch_s, x_t, ei_t, ea_t, batch_t, y=None)

    S = SL if getattr(model, "num_steps", 0) > 0 else S0
    # DGMC returns either dense tensors or objects supporting to_dense()
    if hasattr(S, "to_dense"):
        S = S.to_dense()
    return S  # [ns, nt]


def score_from_S(S: torch.Tensor) -> float:
    """
    Simple graph-level score: mean of per-row maxima.
    This is aligned with DGMC's row-wise matching evaluation (acc/hits use top-k per row).
    """
    if S.numel() == 0:
        return float("nan")
    row_max = S.max(dim=1).values
    return float(row_max.mean().item())


def topk_from_S(
    S: torch.Tensor,
    src: PreparedGraph,
    tgt: PreparedGraph,
    k_top: int,
) -> List[Dict[str, Any]]:
    """
    Returns for each src node the top-k target candidates.
    """
    ns, nt = int(S.size(0)), int(S.size(1))
    if ns == 0:
        return []
    k_eff = int(min(max(k_top, 1), nt)) if nt > 0 else 0

    out: List[Dict[str, Any]] = []
    if k_eff == 0:
        # no target nodes
        for i in range(ns):
            out.append({
                "src_index": i,
                "src_node_id": src.node_ids[i] if i < len(src.node_ids) else None,
                "src_type": src.node_types[i] if i < len(src.node_types) else None,
                "candidates": [],
            })
        return out

    vals, idxs = torch.topk(S, k=k_eff, dim=1)

    vals_cpu = vals.detach().cpu()
    idxs_cpu = idxs.detach().cpu()

    for i in range(ns):
        cand_list = []
        for r in range(k_eff):
            j = int(idxs_cpu[i, r].item())
            cand_list.append({
                "tgt_index": j,
                "tgt_node_id": tgt.node_ids[j] if j < len(tgt.node_ids) else None,
                "tgt_type": tgt.node_types[j] if j < len(tgt.node_types) else None,
                "score": float(vals_cpu[i, r].item()),
            })
        out.append({
            "src_index": i,
            "src_node_id": src.node_ids[i] if i < len(src.node_ids) else None,
            "src_type": src.node_types[i] if i < len(src.node_types) else None,
            "candidates": cand_list,
        })
    return out


def template_label(g: JsonGraph) -> Optional[str]:
    ga = g.get("graph_attrs", {}) or {}
    return str(ga.get("lbs_code") or g.get("label") or g.get("graph_id") or "")


# ============================================================
# Main
# ============================================================

def main() -> None:
    device = torch.device(DEVICE)

    ist_path = _resolve_default(IST_PATH)
    tpl_path = _resolve_default(TEMPLATES_PATH)
    model_path = _resolve_default(MODEL_PATH)
    ensure_dir(OUT_DIR)

    if not ist_path.exists():
        raise FileNotFoundError(f"Ist-Graph JSONL nicht gefunden: {ist_path}")
    if not tpl_path.exists():
        raise FileNotFoundError(f"Template JSONL nicht gefunden: {tpl_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"DGMC model.pt nicht gefunden: {model_path}")

    # Load graphs
    ist_graphs = load_jsonl(ist_path, max_lines=MAX_IST_GRAPHS)
    templates = load_jsonl(tpl_path, max_lines=MAX_TEMPLATES)

    if not templates:
        raise RuntimeError("Keine Templates geladen. Prüfe lbs_soll_graphs_pro.jsonl.")

    print(f"[matcher] device={device} | ist_graphs={len(ist_graphs)} | templates={len(templates)}")
    print(f"[matcher] top-k candidates per template node: K_TOP_MATCHES={K_TOP_MATCHES}")

    # Infer feature dims from one template
    sample = json_graph_to_pyg(templates[0], undirected=UNDIRECTED)
    in_dim = int(sample.x.size(1))
    edge_dim = int(sample.edge_attr.size(1)) if getattr(sample, "edge_attr", None) is not None else 0
    if edge_dim <= 0:
        edge_dim = 1  # best effort

    # Build + load model
    model = build_model(in_dim=in_dim, edge_dim=edge_dim, device=device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[matcher] loaded model: {model_path}")

    # Pre-prepare templates (convert once)
    prepared_templates: List[PreparedGraph] = [prepare_graph(t, device=device) for t in templates]

    # Iterate ist graphs and match
    out_path = OUT_JSONL
    written = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for idx, g_ist in enumerate(ist_graphs, start=1):
            ist_p = prepare_graph(g_ist, device=device)

            tpl_scores: List[Tuple[float, int]] = []

            best_i = -1
            best_score = float("-inf")
            best_S: Optional[torch.Tensor] = None

            # Compare against all templates (keine Reports; nur Ranking + best pick)
            for ti, tpl_p in enumerate(prepared_templates):
                S = dgmc_similarity_matrix(model, src=tpl_p, tgt=ist_p, device=device)  # template -> ist
                sc = score_from_S(S)
                tpl_scores.append((sc, ti))
                if sc > best_score:
                    best_score = sc
                    best_i = ti
                    best_S = S

            # Top-k Template-Ranking exportieren
            tpl_scores.sort(key=lambda x: (-x[0], x[1]))
            top_tpl = tpl_scores[: max(1, int(K_TOP_TEMPLATES))]
            top_templates = []
            for rank, (sc, ti) in enumerate(top_tpl, start=1):
                t = templates[ti]
                top_templates.append({
                    "rank": rank,
                    "template_graph_id": t.get("graph_id"),
                    "template_label": template_label(t),
                    "score": float(sc),
                })

            best_tpl = templates[best_i]
            best_tpl_p = prepared_templates[best_i]

            # Export top-k per template node for the best template only
            if best_S is None:  # pragma: no cover
                topk = []
            else:
                topk = topk_from_S(best_S, src=best_tpl_p, tgt=ist_p, k_top=K_TOP_MATCHES)

            res = {
                "ist_graph_id": g_ist.get("graph_id"),
                "top_templates": top_templates,
                "best_template_graph_id": best_tpl.get("graph_id"),
                "best_template_label": template_label(best_tpl),
                "best_score": float(best_score),
                "topk": topk,
            }
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            written += 1

            if idx % 100 == 0:
                print(f"[matcher] processed {idx}/{len(ist_graphs)}")

    print(f"[matcher] done. wrote {written} results -> {out_path}")


if __name__ == "__main__":
    main()
