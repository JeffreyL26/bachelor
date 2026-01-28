"""
baseline_graph_similarity.py

Wir nutzen exakte Graph Edit Distance (GED)

Similarity score in [0,1]:
      score = 1 - GED / (|V1|+|V2|+|E1|+|E2|)
Niedrigere GED führt damit zu einem höheren Score
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance


JsonGraph = Dict[str, Any]


# ------------------------------
# FUNKTIONEN IMPORTIEREN
# ------------------------------

try:
    from baseline_constraints import (
        _ensure_dir,
        _jsonl_iterieren,
        _jsonl_loader,
        _node_direction,
        _pfad_resolver,
        labeller,
        subset_label_loader,
        optionality_aus_json,
        _lbs_object_direction,
    )
except Exception:
    BASE_DIR = Path(__file__).resolve().parent

    def _default_path(path: Path) -> Path:
        """
        Gibt den Pfad für das Projekt aus
        """
        if path.exists():
            return path
        alt = BASE_DIR / path.name
        if alt.exists():
            return alt
        return path

    def _ensure_dir(p: Path) -> None:
        """
        Sorgt dafür, dass directory existiert
        """
        p.mkdir(parents=True, exist_ok=True)

    def _iter_jsonl(path: Path, max_lines: Optional[int] = None) -> Iterable[JsonGraph]:
        """
        JSONL iterieren
        """
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if max_lines is not None and i > max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _load_jsonl(path: Path, max_lines: Optional[int] = None) -> List[JsonGraph]:
        """
        Lädt JSONL
        """
        return list(_jsonl_iterieren(path, max_lines=max_lines))

    def _node_direction(node: Dict[str, Any]) -> str:
        """
        Das gemeinsame Interface erwartet eine node direction Funktion, aber wir brauchen es hier nicht
        """
        _ = node
        return "unknown"

    # Evaluation Helfer sind hier aus
    def load_bndl2mc_labels(_: Path) -> Dict[Tuple[str, str], str]:
        return {}

    def infer_ground_truth_for_ist(_: JsonGraph, __: Dict[Tuple[str, str], str]) -> Optional[Dict[str, str]]:
        return None

    def load_lbs_optionality_catalog(_: Path) -> Dict[str, Dict[str, Any]]:
        return {}

    def _lbs_object_direction(_: Dict[str, Any]) -> str:
        return "unknown"


# ------------------------------
# NetworkX GRAPH BAUEN
# ------------------------------


def _edge_rel(edge: Dict[str, Any]) -> str:
    """
    Kantentyp extrahieren und vereinheitlichen
    """
    rel = edge.get("rel") or edge.get("type") or edge.get("edge_type") or edge.get("relation")
    if rel is None:
        return "UNKNOWN"
    s = str(rel).strip()
    return s.upper() if s else "UNKNOWN"


def json_to_nx_graph(
    graph: JsonGraph,
    *,
    directed: bool = False,
    enrich_template_dirs: Optional[Dict[str, str]] = None, #TODO
) -> nx.Graph:
    """
    Konvertiert einen JSON Graph in einen NetworkX

    directed:
        If False (default), build an undirected Graph.
    enrich_template_dirs:
        Optional mapping object_code -> direction for template nodes.
        Only used if a node's direction is "unknown".
    """
    G: nx.Graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    for n in nodes:
        node_id = n.get("id")
        if node_id is None:
            continue
        node_id = str(node_id)
        node_type = str(n.get("type") or "UNKNOWN")
        attrs = n.get("attrs") if isinstance(n.get("attrs"), dict) else {}
        obj_code = None
        if isinstance(attrs, dict):
            obj_code = attrs.get("object_code")

        direction = _node_direction(n)
        if direction == "unknown" and enrich_template_dirs and obj_code is not None:
            d2 = enrich_template_dirs.get(str(obj_code))
            if d2:
                direction = str(d2)

        # Nur relevante Attribute
        G.add_node(node_id, type=node_type, direction=direction)

    edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict)]
    for e in edges:
        s = e.get("src")
        t = e.get("dst")
        if s is None or t is None:
            continue
        u = str(s)
        v = str(t)
        if not (G.has_node(u) and G.has_node(v)):
            #Edges ohne src UND dst ignorieren
            continue
        rel = _edge_rel(e)

        #Wenn ein Graph Duplikat-Kanten von beiden Richtungen hat,diese als Set mit einer Edge speichern (falls utiligence mal code fallback braucht)
        if not directed and G.has_edge(u, v):
            prev = G[u][v].get("rel")
            if isinstance(prev, tuple):
                rels = set(prev)
                rels.add(rel)
                G[u][v]["rel"] = tuple(sorted(rels))
            elif isinstance(prev, str):
                if prev != rel:
                    G[u][v]["rel"] = tuple(sorted({prev, rel}))
            else:
                G[u][v]["rel"] = rel
        else:
            G.add_edge(u, v, rel=rel)

    return G


# ------------------------------
# GED KOSTENMODELL
# ------------------------------

@dataclass(frozen=True)
class GedCosts:
    #Kosten pro Verstoß
    insert_node: float = 1.0
    delete_node: float = 1.0
    insert_edge: float = 1.0
    delete_edge: float = 1.0
    type_mismatch: float = 2.0
    direction_mismatch: float = 1.0
    direction_unknown: float = 0.25

    #MENE ignorieren
    ignore_dir_for_types: Tuple[str, ...] = ("MeLo", "NeLo")


def cost_function_maker(costs: GedCosts):
    """
    NetworkX-kompatible Kosten
    """

    def substitute_node_cost(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """
        Kosten für das Ersetzen eines Knotens
        """
        t1 = str(a.get("type") or "UNKNOWN")
        t2 = str(b.get("type") or "UNKNOWN")
        if t1 != t2:
            return float(costs.type_mismatch)

        #Wenn der Knotentyp in der Liste der zu ignorierenden Richtungen steht, dann kostet das Ersetzen durch den gleichen Typ 0
        if t1 in costs.ignore_dir_for_types:
            return 0.0

        d1 = str(a.get("direction") or "unknown")
        d2 = str(b.get("direction") or "unknown")
        if d1 == d2:
            return 0.0
        if d1 == "unknown" or d2 == "unknown":
            return float(costs.direction_unknown)
        return float(costs.direction_mismatch)

    def delete_node_cost(_: Dict[str, Any]) -> float:
        """
        Kosten für die Löschung eines Knotens
        """
        return float(costs.delete_node)

    def insert_node_cost(_: Dict[str, Any]) -> float:
        """
        Kosten für das Einfügen eines Knotens
        """
        return float(costs.insert_node)

    def substitute_edge_cost(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """
        Kosten für das Ersetzen einer Kante
        """
        ra = a.get("rel")
        rb = b.get("rel")
        #Rein in ein Tupel
        if isinstance(ra, str):
            ra_t = (ra,)
        elif isinstance(ra, tuple):
            ra_t = ra
        else:
            ra_t = ("UNKNOWN",)

        if isinstance(rb, str):
            rb_t = (rb,)
        elif isinstance(rb, tuple):
            rb_t = rb
        else:
            rb_t = ("UNKNOWN",)

        return 0.0 if tuple(sorted(ra_t)) == tuple(sorted(rb_t)) else 1.0

    def delete_edge_cost(_: Dict[str, Any]) -> float:
        """
        Kosten für das Löschen einer Kante
        """
        return float(costs.delete_edge)

    def insert_edge_cost(_: Dict[str, Any]) -> float:
        """
        Kosten für das Einsetzen einer Kante
        """
        return float(costs.insert_edge)

    return substitute_node_cost, delete_node_cost, insert_node_cost, substitute_edge_cost, delete_edge_cost, insert_edge_cost


def ged_similarity(
    G1: nx.Graph,
    G2: nx.Graph,
    *,
    costs: GedCosts,
    timeout_s: Optional[float] = None,
) -> Tuple[float, float, float, bool]:
    """
    Gibt (score, ged, ged_norm, timed_out) zurück. Score 0-1, bei 1 identisch

    :param G1: nx.Graph
    :param G2: nx.Graph
    :param costs: GedCosts
    :param timeout_s: Zeitlimit pro einzelner GED-Berechnung
    """
    node_subst, node_del, node_ins, edge_subst, edge_del, edge_ins = cost_function_maker(costs)

    #Wir brauchen einen Normalisierungsfaktor, um aus der absoluten GED-Zahl einen vergleichbaren Wert zu machen
    norm_faktor = float(G1.number_of_nodes() + G2.number_of_nodes() + G1.number_of_edges() + G2.number_of_edges())
    if norm_faktor <= 0:
        return 1.0, 0.0, 0.0, False

    ged = graph_edit_distance(
        G1,
        G2,
        node_subst_cost=node_subst,
        node_del_cost=node_del,
        node_ins_cost=node_ins,
        edge_subst_cost=edge_subst,
        edge_del_cost=edge_del,
        edge_ins_cost=edge_ins,
        timeout=timeout_s,
    )

    if ged is None:
        # Dann maximal anders
        return 0.0, float(norm_faktor), 1.0, True

    ged_f = float(ged)
    ged_norm = max(0.0, min(1.0, ged_f / norm_faktor))
    score = max(0.0, min(1.0, 1.0 - ged_norm))
    return score, ged_f, ged_norm, False


# ------------------------------
# OUTPUT SCHEMA FÜR JSONL
# ------------------------------

def _graph_id(graph: JsonGraph) -> str:
    """
    Gibt die ID des Graphen zurück
    """
    return str(graph.get("graph_id") or "")


def _graph_label(graph: JsonGraph) -> str:
    """
    Gibt das Label des Graphen zurück
    """
    return str(graph.get("label") or graph.get("graph_id") or "")


# ------------------------------
# MAIN
# ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph similarity baseline via exact Graph Edit Distance (GED)")

    p.add_argument("--ist_path", type=str, default=str("data/ist_graphs_all.jsonl"))
    p.add_argument("--templates_path", type=str, default=str("data/lbs_soll_graphs.jsonl"))
    p.add_argument("--out_path", type=str, default=str("data/ged_baseline_matches_3.jsonl"))

    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_ist", type=int, default=None)
    p.add_argument("--max_templates", type=int, default=None)

    #Directed
    p.add_argument(
        "--directed",
        action="store_true",
        help="Treat edges as directed (default: undirected).",
    )

    #Timeout
    p.add_argument(
        "--timeout_s",
        type=float,
        default=None,
        help="Optional timeout per (Ist,Template) GED computation in seconds. "
        "GED may become inexact but runtime becomes robust.",
    )

    # Optionality Block, um Template Node Richtungen evtl. zu befüllen
    p.add_argument(
        "--lbs_json_dir",
        type=str,
        default=str("data/lbs_templates"),
        help="Directory containing LBS JSONs with _lbs_optionality (used for template directions).",
    )

    # Standardmäßige Evaluierung auf gelabelten Subset
    p.add_argument(
        "--bndl2mc_path",
        type=str,
        default=str("data/training_data/BNDL2MC.csv"),
        help="Optional BNDL2MC.csv for evaluation (default: data/training_data/BNDL2MC.csv). Use '' to disable.",
    )

    #Kostenfunktion einstellen
    p.add_argument("--type_mismatch_cost", type=float, default=2.0)
    p.add_argument("--dir_mismatch_cost", type=float, default=1.0)
    p.add_argument("--dir_unknown_cost", type=float, default=0.25)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ist_pfad = _pfad_resolver(Path(args.ist_path))
    soll_pfad = _pfad_resolver(Path(args.templates_path))
    out_path = Path(args.out_path)
    _ensure_dir(out_path.parent)

    if not ist_pfad.exists():
        raise FileNotFoundError(f"Ist graphs JSONL not found: {ist_pfad}")
    if not soll_pfad.exists():
        raise FileNotFoundError(f"Template graphs JSONL not found: {soll_pfad}")

    ist_graphs = _jsonl_loader(ist_pfad, max_lines=args.max_ist)
    templates = _jsonl_loader(soll_pfad, max_lines=args.max_templates)
    if not templates:
        raise RuntimeError("No templates loaded.")

    # Optionalitäten-Block laden (für enrichment der Template directions)
    lbs_json_dir = _pfad_resolver(Path(args.lbs_json_dir))
    opt_catalog: Dict[str, Dict[str, Any]] = {}
    if lbs_json_dir.exists() and lbs_json_dir.is_dir():
        opt_catalog = optionality_aus_json(lbs_json_dir)
    else:
        print(f"[ged][warn] lbs_json_dir not found / not a directory: {lbs_json_dir} (template dir enrichment disabled)")

    # Evaluierungslabel
    paar_zu_mcid: Dict[Tuple[str, str], str] = {}
    bndl_argumente = str(args.bndl2mc_path or "").strip()
    if bndl_argumente:
        bndl_pfad = _pfad_resolver(Path(bndl_argumente))
        if bndl_pfad.exists():
            paar_zu_mcid = subset_label_loader(bndl_pfad)
            print(f"[ged] Loaded BNDL2MC pairs: {len(paar_zu_mcid)} from {bndl_pfad}")
        else:
            print(f"[ged][warn] BNDL2MC.csv not found at: {bndl_pfad} (evaluation disabled)")

    print(
        f"[ged] ist_graphs={len(ist_graphs)} | templates={len(templates)} | top_k={args.top_k} | "
        f"directed={bool(args.directed)} | timeout_s={args.timeout_s}"
    )

    costs = GedCosts(
        type_mismatch=float(args.type_mismatch_cost),
        direction_mismatch=float(args.dir_mismatch_cost),
        direction_unknown=float(args.dir_unknown_cost),
    )

    # Für jedes Template einen NetworkX-Graphen bauen
    # Mapping für Richtung falls Template Richtung unknown und enrich an
    soll_nx: List[nx.Graph] = []
    soll_meta: List[Dict[str, Any]] = []
    for t in templates:
        label = _graph_label(t)
        enrich_dirs: Optional[Dict[str, str]] = None

        eintrag = opt_catalog.get(label)
        if isinstance(eintrag, dict):
            opt = eintrag.get("opt")
            if isinstance(opt, dict):
                lbs_objs = opt.get("lbs_objects")
                if isinstance(lbs_objs, list):
                    # object_code zu direction mappen
                    tmp: Dict[str, str] = {}
                    for o in lbs_objs:
                        if not isinstance(o, dict):
                            continue
                        oc = o.get("object_code")
                        if oc is None:
                            continue
                        d = _lbs_object_direction(o)
                        if d and d != "unknown":
                            tmp[str(oc)] = str(d)
                    enrich_dirs = tmp if tmp else None

        #NetworkX Template Graph Gt
        Gt = json_to_nx_graph(t, directed=bool(args.directed), enrich_template_dirs=enrich_dirs)
        soll_nx.append(Gt)
        soll_meta.append(
            {
                "template_graph_id": _graph_id(t),
                "template_label": label,
                "n_nodes": int(Gt.number_of_nodes()),
                "n_edges": int(Gt.number_of_edges()),
            }
        )

    # Evaluierungszähler
    eval_total = 0
    eval_top1 = 0
    eval_top3 = 0

    top1_distribution: Counter = Counter()
    #Falls wir limited Zeit genutzt haben
    timed_out_pairs = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for g in ist_graphs:
            #NetworkX Graph für Ist-Graphen Gi
            Gi = json_to_nx_graph(g, directed=bool(args.directed), enrich_template_dirs=None)
            ist_id = _graph_id(g)

            scored: List[Tuple[float, float, int, float, bool]] = []
            # Tupel aus (score_desc, ged, idx, ged_norm, timed_out)
            for i, Gt in enumerate(soll_nx):
                score, ged, ged_norm, to = ged_similarity(Gi, Gt, costs=costs, timeout_s=args.timeout_s)
                if to:
                    timed_out_pairs += 1
                scored.append((score, ged, i, ged_norm, to))

            scored.sort(key=lambda x: (-x[0], x[1], x[2]))
            top_k = max(1, int(args.top_k))
            top = scored[:top_k]

            top_templates: List[Dict[str, Any]] = []
            for rank, (score, ged, idx, ged_norm, to) in enumerate(top, start=1):
                meta = soll_meta[idx]
                top_templates.append(
                    {
                        "rank": rank,
                        "template_graph_id": meta["template_graph_id"],
                        "template_label": meta["template_label"],
                        "score": float(score),
                        "ged": float(ged),
                        "ged_norm": float(ged_norm),
                        "timed_out": bool(to),
                        "sizes": {
                            "ist_nodes": int(Gi.number_of_nodes()),
                            "ist_edges": int(Gi.number_of_edges()),
                            "tpl_nodes": int(meta["n_nodes"]),
                            "tpl_edges": int(meta["n_edges"]),
                        },
                        "cost_model": {
                            "type_mismatch": float(costs.type_mismatch),
                            "dir_mismatch": float(costs.direction_mismatch),
                            "dir_unknown": float(costs.direction_unknown),
                            "ignore_dir_for_types": list(costs.ignore_dir_for_types),
                        },
                    }
                )

            if top_templates:
                top1_distribution[top_templates[0]["template_label"]] += 1

            #Evaluationsblock
            ground_truth = labeller(g, paar_zu_mcid) if paar_zu_mcid else None
            if ground_truth and ground_truth.get("template_label"):
                eval_total += 1
                ground_truth_label = ground_truth["template_label"]
                pred1 = top_templates[0]["template_label"] if top_templates else ""
                if pred1 == ground_truth_label:
                    eval_top1 += 1
                pred_labels = [x["template_label"] for x in top_templates[:3]]
                if ground_truth_label in pred_labels:
                    eval_top3 += 1

            out_obj = {
                "ist_graph_id": ist_id,
                "top_templates": top_templates,
                "ground_truth": ground_truth,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"[ged] wrote: {out_path}")

    # Matching-Verteilung
    if top1_distribution:
        total = sum(top1_distribution.values())
        print("[ged] top1 prediction distribution (all Ist graphs):")
        for lbl, cnt in top1_distribution.most_common():
            pct = 100.0 * float(cnt) / float(total)
            print(f"  - {lbl}: {cnt} ({pct:.3f}%)")

    if timed_out_pairs > 0:
        print(f"[ged][warn] timed_out_pairs={timed_out_pairs} (consider increasing --timeout_s or leaving it unset for exact GED)")

    if eval_total > 0:
        print(
            "[ged] evaluation on labelled subset | "
            f"n={eval_total} | hits@1={eval_top1/eval_total:.3f} | hits@3={eval_top3/eval_total:.3f}"
        )


if __name__ == "__main__":
    main()
