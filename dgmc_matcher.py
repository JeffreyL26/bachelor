from __future__ import annotations

"""
dgmc_matcher.py

Matcht Ist-Graphen mit den Templates. Gibt Top-k aus. Für das Top-1 auch die jeweiligen Knoten-Korrespondenzen.
Evaluierung auf gelabelten Subset.
"""

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

#DGMC Import
try:
    from dgmc.models import DGMC
except Exception:
    from torch_geometric.nn import DGMC

from torch_geometric.nn import GINEConv

import graph_pipeline as gp


# ------------------------------
# MODELL DEFINIEREN
# ------------------------------

class EdgeAwareGINE(nn.Module):
    """
    Gleicher EdgeAwareGINE wie in @dgmc_template_training.py, keine Zeit mehr auszulagern
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_dim = int(edge_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(c_in, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None: #wie in @dgmc_template_training.py
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.norms:
            bn.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(                        #wie in @dgmc_template_training.py
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))

        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin_out(x)


# ------------------------------
# IO helpers
# ------------------------------

TGraph = Dict[str, Any]


def _lade_pfad(pfad: Path) -> Path:
    """
    Pfad laden
    """
    if pfad.exists():
        return pfad
    alt = Path(__file__).resolve().parent / pfad.name
    if alt.exists():
        return alt
    return pfad


def jsonl_iter(pfad: Path, max_lines: Optional[int] = None) -> Iterable[TGraph]:
    """
    JSONL durchlaufen
    """
    with pfad.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_lines is not None and line_no > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON in {pfad} at line {line_no}: {e}") from e


def load_jsonl(pfad: Path, max_lines: Optional[int] = None) -> List[TGraph]:
    """
    JSONL aus Pfad laden
    """
    return list(jsonl_iter(pfad, max_lines=max_lines))


def ensure_dir(p: Path) -> None:
    """
    Verzeichnis erstellen (wenn nötig)
    """
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------
# Evaluierung
# ------------------------------

LBS_CODES: Dict[str, str] = {
    "S_A1_A2": "9992000000042",
    "S_C3": "9992000000175",
    "S_A001": "9992000000026",
}


def label_laden(bndl2mc_path: Path) -> Dict[Tuple[str, str], str]:
    """
    Labelt die Evaluierungsgraphen
    """
    mapping: Dict[Tuple[str, str], str] = {}
    with bndl2mc_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(";")
        spalten = {name: i for i, name in enumerate(header)}
        benoetigte_spalten = {"Marktlokation", "Messlokation", "MCID"}
        if not benoetigte_spalten.issubset(spalten.keys()):
            raise ValueError(
                f"BNDL2MC header missing required columns {sorted(benoetigte_spalten)}. "
                f"Got: {header}"
            )

        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) <= max(spalten.values()):
                continue

            # MaLo manchmal numerisch gespeichert, casten
            try:
                malo = str(int(parts[spalten["Marktlokation"]]))
            except Exception:
                malo = str(parts[spalten["Marktlokation"]]).strip()

            melo = str(parts[spalten["Messlokation"]]).strip()
            mcid = str(parts[spalten["MCID"]]).strip()

            if not malo or not melo or not mcid:
                continue

            mapping[(malo, melo)] = mcid

    return mapping


def infer_ground_truth_for_ist(
    graph: TGraph,
    paar_label: Dict[Tuple[str, str], str],
) -> Optional[Dict[str, str]]:
    """
    Wenn ein Graph in den gelabelten Graphen auftaucht, Ground-Truth ausgeben
    """
    #Wir prüfen für Malo-Melo-Paare, denn diese sind in den Spalten der Label-Tabelle fix gegeben
    nodes = graph.get("nodes") or []
    malos = [
        str(n.get("id"))
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "MaLo" and n.get("id") is not None
    ]
    melos = [
        str(n.get("id"))
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "MeLo" and n.get("id") is not None
    ]

    found: Optional[Tuple[str, str]] = None
    found_mcid: Optional[str] = None
    for malo in malos:
        for melo in melos:
            mcid = paar_label.get((malo, melo))
            if mcid:
                found = (malo, melo)
                found_mcid = mcid
                break
        if found_mcid:
            break

    if not found_mcid:
        return None

    return {
        "mcid": found_mcid,
        "template_label": LBS_CODES.get(found_mcid, ""),
        "malo": found[0] if found else "",
        "melo": found[1] if found else "",
    }


# ------------------------------
# GESPEICHERTES MODELL LADEN
# ------------------------------

def _best_checkpoint_modell_laden(pfad: Path, device: torch.device) -> Dict[str, Any]:
    """
    Lädt das gespeicherte Modell (bester Checkpoint während des Trainings)
    """
    obj = torch.load(pfad, map_location=device)

    # {"epoch": ..., "model_state": ..., "optimizer_state": ..., "args": ..., "in_channels": ..., "edge_dim": ...}
    if isinstance(obj, dict) and "model_state" in obj:
        return obj

    # Als Fallback leeres Modell akzeptieren (nur aus Error-Zwecken)
    if isinstance(obj, dict):
        return {"model_state": obj, "args": {}, "in_channels": None, "edge_dim": None}

    raise ValueError(
        "Unsupported checkpoint format. Expected a dict with key 'model_state'."
    )


def _dimensionen_aus_graph(sample_graph: TGraph, undirected: bool) -> Tuple[int, int]:
    d = gp.json_graph_to_pyg(sample_graph, undirected=undirected)
    in_channels = int(d.x.size(-1))
    edge_attr = getattr(d, "edge_attr", None)
    if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.size(1) > 0:
        edge_dim = int(edge_attr.size(1))
    else:
        # graph_pipeline.py nutzt 4 + 1 unknown => 5
        edge_dim = 5
    return in_channels, edge_dim


def modell_bauen(
    checkpoint: Dict[str, Any],
    *,
    sample_graph: TGraph,
    undirected: bool,
    device: torch.device,
    override_num_steps: Optional[int] = None,
    override_k: Optional[int] = None,
    override_detach: Optional[bool] = None,
) -> DGMC:
    checkpoint_args: Dict[str, Any] = dict(checkpoint.get("args") or {})
    """
    Baut Modell aus besagtem Checkpoint
    """

    # Dimensionen
    in_channels = checkpoint.get("in_channels")
    edge_dim = checkpoint.get("edge_dim")
    if (
        not isinstance(in_channels, int)
        or not isinstance(edge_dim, int)
        or in_channels <= 0
        or edge_dim <= 0
    ):
        in_channels, edge_dim = _dimensionen_aus_graph(sample_graph, undirected=undirected)

    # Architektur Parameter (default: 64,64,3,0.0)
    hidden_channels = int(checkpoint_args.get("hidden_channels", 64))
    out_channels = int(checkpoint_args.get("out_channels", 64))
    num_layers = int(checkpoint_args.get("num_layers", 3))
    dropout = float(checkpoint_args.get("dropout", 0.0))

    num_steps = int(checkpoint_args.get("num_steps", 10))
    k = int(checkpoint_args.get("k", -1))                           #-1 Dense: Betrachtet alle möglichen Target-Knoten als Kandidaten. >=1: Beschränkt auf bese k Kandidaten pro Source-Knoten
    detach = bool(checkpoint_args.get("detach", False))

    if override_num_steps is not None:
        num_steps = int(override_num_steps)
    if override_k is not None:
        k = int(override_k)
    if override_detach is not None:
        detach = bool(override_detach)

    psi_1 = EdgeAwareGINE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=edge_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    psi_2 = EdgeAwareGINE(
        in_channels=out_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=edge_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    model = DGMC(
        psi_1=psi_1,
        psi_2=psi_2,
        num_steps=num_steps,
        k=k,
        detach=detach,
    ).to(device)

    state_dict = checkpoint["model_state"]
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint['model_state'] must be a state_dict dict")

    #Strict=True: Keine Abweichungen zwischen gespeicherten Modell und aktuellem Modellcode akzeptieren
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ------------------------------
# MATCHING LOGIK
# ------------------------------

@dataclass
class VollerGraph:
    graph_json: TGraph
    data: Any  # PyG Data
    node_ids: List[Any]
    node_types: List[Any]


def graph_vorbereiten(graph: TGraph, *, undirected: bool, device: torch.device) -> VollerGraph:
    """
    Aus TGraph einen @VollerGraph machen
    """
    d = gp.json_graph_to_pyg(graph, undirected=undirected).to(device)
    node_ids = list(getattr(d, "node_ids", []))
    node_types = list(getattr(d, "node_types", []))
    return VollerGraph(graph_json=graph, data=d, node_ids=node_ids, node_types=node_types)


@torch.no_grad()
def dgmc_similarity_matrix(
    model: DGMC,
    *,
    source: VollerGraph,
    target: VollerGraph,
    device: torch.device,
) -> torch.Tensor:
    """
    Vollständige Matrix S berechnen. Similarity-Score für jede Source-Node i und Target-Node j
    """

    x_source = source.data.x
    edge_indizes_source = source.data.edge_index
    edge_attribute_source = getattr(source.data, "edge_attr", None)

    x_target = target.data.x
    edge_indizes_target = target.data.edge_index
    edge_attribute_target = getattr(target.data, "edge_attr", None)

    ns = int(x_source.size(0))
    nt = int(x_target.size(0))

    source_batch = torch.zeros(ns, dtype=torch.long, device=device)
    target_batch = torch.zeros(nt, dtype=torch.long, device=device)
    #Korrespondenzmatrizen
    S0, SL = model(x_source, edge_indizes_source, edge_attribute_source, source_batch, x_target, edge_indizes_target, edge_attribute_target, target_batch, y=None)

    #Wenn refined Matrix SL verfügbar ist, dann natürlich die nutzen
    S = SL if SL is not None else S0

    # Wenn Matrix nicht voll ausgefüllt ist, wird sie in eine umgewandelt
    if hasattr(S, "to_dense"):
        S = S.to_dense()

    # Wenn Padding angewendet wurde, dann auf richtiges nt zuschneiden
    if isinstance(S, torch.Tensor) and S.dim() == 2 and S.size(1) > nt:
        S = S[:, :nt]

    if not isinstance(S, torch.Tensor) or S.dim() != 2:
        raise RuntimeError("DGMC produced an unexpected score matrix type/shape")

    return S


def score_mean_rowmax(S: torch.Tensor) -> float:
    """
    Berechnet einen Graph-Score aus der Similarity-Matrix S,
    indem für jede Source-Zeile der größten Ähnlichkeitswert genommen wird (beste Zuordnung pro Source-Knoten) und diese Maxima dann mittelt
    Ist S leer, gibt sie -unendlich zurück.
    """
    if S.numel() == 0:
        return float("-inf")
    return float(S.max(dim=1).values.mean().item())


@torch.no_grad()
def paar_scoren(
    model: DGMC,
    *,
    template: VollerGraph,
    ist: VollerGraph,
    device: torch.device,
    score_mode: str,
) -> Tuple[float, torch.Tensor]:
    """
    Return (score, S_template_to_ist)
    """
    S_template2instance = dgmc_similarity_matrix(model, source=template, target=ist, device=device)     #Ähnlichkeitsmatrix zwischen Template und Ist-Graph
    s1 = score_mean_rowmax(S_template2instance)                                                         #Score als Zahl

    if score_mode == "mean_rowmax":
        return s1, S_template2instance

    if score_mode == "mean_rowmax_symmetric":
        S_instance2template = dgmc_similarity_matrix(model, source=ist, target=template, device=device)
        s2 = score_mean_rowmax(S_instance2template)
        return 0.5 * (s1 + s2), S_template2instance

    raise ValueError(f"Unknown score_mode: {score_mode}")


def topk_node(
    S: torch.Tensor,
    *,
    source: VollerGraph,
    target: VollerGraph,
    k_top: int,
) -> List[Dict[str, Any]]:
    """
    Für jede Source Node, die Top-k Kandidaten aus Target holen
    """
    if S.dim() != 2:
        return []

    source_nodes, target_nodes = int(S.size(0)), int(S.size(1))
    if source_nodes <= 0:
        return []

    k_anzahl = int(min(max(int(k_top), 1), target_nodes)) if target_nodes > 0 else 0

    out: List[Dict[str, Any]] = []
    if k_anzahl == 0:
        for sn in range(source_nodes):
            out.append(
                {
                    "src_index": sn,
                    "src_node_id": source.node_ids[sn] if sn < len(source.node_ids) else None,
                    "src_type": source.node_types[sn] if sn < len(source.node_types) else None,
                    "candidates": [],
                }
            )
        return out

    #Für jede Zeile von S (jeder Source-Knoten)
    top_k_scores, target_node_indizes = torch.topk(S, k=k_anzahl, dim=1)
    topk_auf_cpu = top_k_scores.detach().cpu()
    target_node_auf_cpu = target_node_indizes.detach().cpu()

    for sn in range(source_nodes):
        kandidaten_liste = []
        for tk in range(k_anzahl):
            j = int(target_node_auf_cpu[sn, tk].item())
            kandidaten_liste.append(
                {
                    "tgt_index": j,
                    "tgt_node_id": target.node_ids[j] if j < len(target.node_ids) else None,
                    "tgt_type": target.node_types[j] if j < len(target.node_types) else None,
                    "score": float(topk_auf_cpu[sn, tk].item()),
                }
            )
        out.append(
            {
                "src_index": sn,
                "src_node_id": source.node_ids[sn] if sn < len(source.node_ids) else None,
                "src_type": source.node_types[sn] if sn < len(source.node_types) else None,
                "candidates": kandidaten_liste,
            }
        )

    return out


def template_label(graph: TGraph) -> Optional[str]:
    """
    Gibt ID zurück
    """
    graph_attribute = graph.get("graph_attrs", {}) or {}
    x = graph_attribute.get("lbs_code") or graph.get("label") or graph.get("graph_id")
    return str(x) if x is not None else None


# ------------------------------
# MAIN
# ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match Ist graphs to templates with DGMC.")

    p.add_argument("--ist_path", type=str, default=os.path.join("data", "ist_graphs_all.jsonl"))
    p.add_argument("--templates_path", type=str, default=os.path.join("data", "lbs_soll_graphs.jsonl"))
    #p.add_argument("--checkpoint", type=str, default=os.path.join("data", "dgmc_partial.pt"))
    p.add_argument("--checkpoint", type=str, default=os.path.join("data", "dgmc_perm.pt"))
    #p.add_argument("--out_path", type=str, default=os.path.join("runs", "dgmc_matches_top3_partial.jsonl"))
    p.add_argument("--out_path", type=str, default=os.path.join("runs", "dgmc_matches_top3_perm.jsonl"))


    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--undirected", action="store_true", help="Use undirected edges (should match training).")
    p.add_argument("--directed", action="store_true", help="Use directed edges (overrides --undirected).")

    p.add_argument("--top_templates", type=int, default=3, help="How many best templates to return per Ist graph.")
    p.add_argument("--top_matches", type=int, default=3, help="Top-k node candidates per template node (best template).")

    p.add_argument(
        "--score_mode",
        type=str,
        default="mean_rowmax",
        choices=["mean_rowmax", "mean_rowmax_symmetric"],
        help="Graph-level ranking from DGMC similarity matrix.",
    )

    #Nur bestimmte Menge nutzen (Debug, vielleicht noch vor Abgabe rausnehmen)
    p.add_argument("--max_ist", type=int, default=0, help="Process only the first N ist graphs (0 = all).")
    p.add_argument("--max_templates", type=int, default=0, help="Use only the first N templates (0 = all).")

    #DGMC Trainingsparameter kann man damit überschreiben
    p.add_argument("--override_num_steps", type=int, default=-999)
    p.add_argument("--override_k", type=int, default=-999)
    p.add_argument("--override_detach", action="store_true")
    p.add_argument("--no_override_detach", action="store_true")

    # Evaluierung
    p.add_argument(
        "--bndl2mc_path",
        type=str,
        default=str(os.path.join("data", "training_data", "BNDL2MC.csv")),
        help="BNDL2MC.csv for evaluation (default: data/training_data/BNDL2MC.csv). Use '' to disable.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    #Default undirected
    undirected = True
    if args.directed:
        undirected = False
    elif args.undirected:
        undirected = True

    device = torch.device(args.device)

    ist_pfad = _lade_pfad(Path(args.ist_path))
    template_pfadh = _lade_pfad(Path(args.templates_path))
    modell_checkpoint_pfad = _lade_pfad(Path(args.checkpoint))
    out_path = Path(args.out_path)

    if not ist_pfad.exists():
        raise FileNotFoundError(f"Ist-Graph JSONL not found: {ist_pfad}")
    if not template_pfadh.exists():
        raise FileNotFoundError(f"Template JSONL not found: {template_pfadh}")
    if not modell_checkpoint_pfad.exists():
        raise FileNotFoundError(f"DGMC checkpoint not found: {modell_checkpoint_pfad}")

    ensure_dir(out_path.parent)

    max_ist = None if int(args.max_ist) <= 0 else int(args.max_ist)
    max_templates = None if int(args.max_templates) <= 0 else int(args.max_templates)

    if max_ist is not None:
        print(f"[dgmc][note] --max_ist={max_ist} : Not all Ist-Graphs will be processed.")
    if max_templates is not None:
        print(f"[dgmc][note] --max_templates={max_templates} : Not all Template-Graphs will be processed.")

    templates = load_jsonl(template_pfadh, max_lines=max_templates)
    if not templates:
        raise RuntimeError("No templates loaded (templates JSONL empty or invalid).")

    # Evaluierungs-Infos
    paar_mcid: Dict[Tuple[str, str], str] = {}
    bndl_arg = str(args.bndl2mc_path or "").strip()
    if bndl_arg:
        bndl_path = _lade_pfad(Path(bndl_arg))
        if bndl_path.exists():
            paar_mcid = label_laden(bndl_path)
            print(f"[dgmc] Loaded BNDL2MC pairs: {len(paar_mcid)} from {bndl_path}")
        else:
            print(f"[dgmc][warn] BNDL2MC.csv not found at: {bndl_path} (evaluation disabled)")

    # Build und Modell laden
    best_modell_checkpoint = _best_checkpoint_modell_laden(modell_checkpoint_pfad, device=device)
    #Überschreiben, wenn festgelegt
    override_num_steps = None if args.override_num_steps == -999 else int(args.override_num_steps)
    override_k = None if args.override_k == -999 else int(args.override_k)
    override_detach: Optional[bool] = None
    if args.override_detach and args.no_override_detach:
        raise ValueError("Use only one of --override_detach / --no_override_detach")
    if args.override_detach:
        override_detach = True
    if args.no_override_detach:
        override_detach = False

    model = modell_bauen(
        best_modell_checkpoint,
        sample_graph=templates[0],
        undirected=undirected,
        device=device,
        override_num_steps=override_num_steps,
        override_k=override_k,
        override_detach=override_detach,
    )

    # Templates vorbereiten
    prepared_templates: List[VollerGraph] = [
        graph_vorbereiten(t, undirected=undirected, device=device) for t in templates
    ]

    # Evaluation Zähler
    eval_total = 0
    eval_top1 = 0
    eval_top3 = 0

    # Verteilung der Matches
    pred_top1_dist: Counter = Counter()

    # Iterieren
    written = 0
    ist_iter = jsonl_iter(ist_pfad, max_lines=max_ist)

    print(
        f"[dgmc] device={device} | undirected={undirected} | ist_graphs={'ALL' if max_ist is None else max_ist} "
        f"| templates={len(prepared_templates)} | top_k={args.top_templates} | top_matches={args.top_matches} "
        f"| score_mode={args.score_mode}"
    )

    with out_path.open("w", encoding="utf-8") as f_out:
        for index_ist, ist_g in enumerate(ist_iter, start=1):
            ist_p = graph_vorbereiten(ist_g, undirected=undirected, device=device)

            template_scores: List[Tuple[float, int]] = []

            best_index = -1
            best_score = float("-inf")
            best_S_template2instance: Optional[torch.Tensor] = None

            for template_index, prepared_template in enumerate(prepared_templates):
                sc, S_t2i = paar_scoren(
                    model,
                    template=prepared_template,
                    ist=ist_p,
                    device=device,
                    score_mode=args.score_mode,
                )

                template_scores.append((float(sc), template_index))
                if sc > best_score:
                    best_score = float(sc)
                    best_index = template_index
                    best_S_template2instance = S_t2i

            template_scores.sort(key=lambda x: (-x[0], x[1]))
            top_n = max(1, int(args.top_templates))
            top_templates = template_scores[:top_n]

            top_templates_out = []
            for rank, (sc, template_index) in enumerate(top_templates, start=1):
                t = templates[template_index]
                top_templates_out.append(
                    {
                        "rank": rank,
                        "template_graph_id": t.get("graph_id"),
                        "template_label": template_label(t),
                        "score": float(sc),
                    }
                )

            # Verteilung der Templates über Top-1 Picks
            if top_templates_out:
                pred_top1_dist[str(top_templates_out[0].get("template_label") or "")] += 1

            bestes_template = templates[best_index]
            bestes_template_prepared = prepared_templates[best_index]

            # Bestes Template Top-k Nodes
            if best_S_template2instance is None:
                topk = []
            else:
                topk = topk_node(
                    best_S_template2instance, source=bestes_template_prepared, target=ist_p, k_top=int(args.top_matches)
                )

            # Ground-Truth zur Evaluierung
            ground_truth = infer_ground_truth_for_ist(ist_g, paar_mcid) if paar_mcid else None
            if ground_truth and ground_truth.get("template_label"):
                eval_total += 1
                ground_truth_label = str(ground_truth["template_label"])
                top_prediction = str(top_templates_out[0].get("template_label") or "") if top_templates_out else ""
                if top_prediction == ground_truth_label:
                    eval_top1 += 1
                prediction_labels = [str(x.get("template_label") or "") for x in top_templates_out[:3]]
                if ground_truth_label in prediction_labels:
                    eval_top3 += 1

            res = {
                "ist_graph_id": ist_g.get("graph_id"),
                "top_templates": top_templates_out,
                "ground_truth": ground_truth,
                # DGMC-spezifisch für Debugging
                "best_template_graph_id": bestes_template.get("graph_id"),
                "best_template_label": template_label(bestes_template),
                "best_score": float(best_score),
                "topk": topk,
            }

            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            written += 1

            if index_ist % 50 == 0:
                print(f"[dgmc] processed {index_ist} ist graphs")

    print(f"[dgmc] wrote: {out_path}")

    # Ranking Verteilung
    anzahl_predictions = sum(int(v) for v in pred_top1_dist.values())
    if anzahl_predictions > 0:
        print("[dgmc] top1 prediction distribution (processed Ist graphs):")
        for label_soll, anzahl in pred_top1_dist.most_common():
            prozentsatz = 100.0 * float(anzahl) / float(anzahl_predictions)
            print(f"  - {label_soll}: {anzahl} ({prozentsatz:.3f}%)")

    # Evaluierungs-Report
    if eval_total > 0:
        print(
            "[dgmc] evaluation on labeled subset | "
            f"n={eval_total} | hits@1={eval_top1/eval_total:.3f} | hits@3={eval_top3/eval_total:.3f}"
        )
    else:
        if paar_mcid:
            print(
                "[dgmc][warn] evaluation skipped: no labeled graphs found that map to known template labels."
            )
        else:
            print("[dgmc] evaluation skipped (no BNDL2MC loaded).")

    print(f"[dgmc] done. wrote {written} results")


if __name__ == "__main__":
    main()