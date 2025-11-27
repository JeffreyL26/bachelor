from __future__ import annotations
import json
from typing import Dict, Any, List, Optional, Iterable
import torch
from torch_geometric.data import Data

# -----------------------------
# Konfiguration: Kodierungen
# -----------------------------

# Knotentypen in deinen Graphen
NODE_TYPES = {
    "MaLo": 0,
    "MeLo": 1,
    "TR":   2,
    "NeLo": 3,
}
NUM_NODE_TYPES = len(NODE_TYPES)

# Richtung der MaLo
DIRECTIONS = {
    "consumption": 0,
    "generation":  1,
    # alles andere → unknown
}
DIR_UNKNOWN_INDEX = len(DIRECTIONS)
NUM_DIRECTIONS = len(DIRECTIONS) + 1  # + unknown

# MeLo-Funktion in Templates
MELO_FUNCTIONS = {
    "N": 0,
    "H": 1,
    "D": 2,
    "S": 3,
    # alles andere → unknown
}
MELO_FUNC_UNKNOWN_INDEX = len(MELO_FUNCTIONS)
NUM_MELO_FUNCTIONS = len(MELO_FUNCTIONS) + 1

# Spannungsebene aus Ist-Daten (Spannungsebene E05/E06)
# Falls du später andere Codes siehst, einfach ergänzen.
VOLTAGE_LEVELS = {
    "E05": 0,
    "E06": 1,
}
VOLT_UNKNOWN_INDEX = len(VOLTAGE_LEVELS)
NUM_VOLTAGE_LEVELS = len(VOLTAGE_LEVELS) + 1

# Kanten-Typen
EDGE_TYPES = {
    "MEMA": 0,
    "METR": 1,
    "MENE": 2,
    "MEME": 3,
}
EDGE_UNKNOWN_INDEX = len(EDGE_TYPES)
NUM_EDGE_TYPES = len(EDGE_TYPES) + 1


# -----------------------------
# Hilfsfunktionen
# -----------------------------

def _one_hot(index: int, length: int) -> List[float]:
    v = [0.0] * length
    if 0 <= index < length:
        v[index] = 1.0
    return v


def _encode_node_features(node: Dict[str, Any]) -> List[float]:
    """
    Kodiert einen JSON-Knoten aus deinen Graphen in einen Feature-Vektor.

    Erwartetes Node-Format:
    {
        "id": "<string>",
        "type": "MaLo" | "MeLo" | "TR" | "NeLo",
        "attrs": { ... }
    }
    """
    ntype = node.get("type")
    attrs = node.get("attrs", {}) or {}

    # 1) node_type one-hot
    type_idx = NODE_TYPES.get(ntype, None)
    if type_idx is None:
        # Unbekannter Typ → alles 0, optional könntest du hier auch erweitern
        type_vec = [0.0] * NUM_NODE_TYPES
    else:
        type_vec = _one_hot(type_idx, NUM_NODE_TYPES)

    # 2) direction one-hot (für MaLo, bei anderen egal)
    raw_dir = attrs.get("direction")
    dir_idx = DIRECTIONS.get(raw_dir, DIR_UNKNOWN_INDEX)
    dir_vec = _one_hot(dir_idx, NUM_DIRECTIONS)

    # 3) MeLo-Funktion (nur bei MeLo wirklich belegt)
    raw_fn = attrs.get("function") if ntype == "MeLo" else None
    fn_idx = MELO_FUNCTIONS.get(raw_fn, MELO_FUNC_UNKNOWN_INDEX)
    fn_vec = _one_hot(fn_idx, NUM_MELO_FUNCTIONS)

    # 4) voltage_level (nur in Ist-MeLo vorhanden)
    raw_volt = attrs.get("voltage_level") if ntype == "MeLo" else None
    volt_idx = VOLTAGE_LEVELS.get(raw_volt, VOLT_UNKNOWN_INDEX)
    volt_vec = _one_hot(volt_idx, NUM_VOLTAGE_LEVELS)

    # 5) level (nur in Templates gesetzt; bei Ist default 0.0)
    level = float(attrs.get("level", 0.0))

    return type_vec + dir_vec + fn_vec + volt_vec + [level]


def _encode_edge_attr(rel: str) -> List[float]:
    """
    Kodiert Kanten-Typ (rel) als One-Hot-Feature.
    Erwartete Werte: "MEMA", "METR", "MENE", "MEME".
    """
    idx = EDGE_TYPES.get(rel, EDGE_UNKNOWN_INDEX)
    return _one_hot(idx, NUM_EDGE_TYPES)


# -----------------------------
# JSON-Graph -> PyG Data
# -----------------------------

def json_graph_to_pyg(
    g: Dict[str, Any],
    undirected: bool = True,
) -> Data:
    """
    Konvertiert einen Graphen im JSON-Format aus deinen Dateien (ist_graphs_small.jsonl
    oder lbs_templates.jsonl) in ein torch_geometric.data.Data-Objekt.

    Erwartetes dict-Format:
    {
        "graph_id": str,
        "label": Optional[str],
        "nodes": [
            {"id": str, "type": str, "attrs": {...}},
            ...
        ],
        "edges": [
            {"src": str, "dst": str, "rel": str},
            ...
        ],
        "graph_attrs": {...}
    }
    """

    nodes = g.get("nodes", [])
    edges = g.get("edges", [])

    # Map Node-ID -> Index
    id_to_idx: Dict[str, int] = {n["id"]: i for i, n in enumerate(nodes)}

    # Node-Features
    x_list: List[List[float]] = [_encode_node_features(n) for n in nodes]
    x = torch.tensor(x_list, dtype=torch.float)

    # Kanten & Edge-Features
    edge_index_list: List[List[int]] = []
    edge_attr_list: List[List[float]] = []

    for e in edges:
        src_id = e.get("src")
        dst_id = e.get("dst")
        rel = e.get("rel")

        if src_id not in id_to_idx or dst_id not in id_to_idx:
            # Inkonsistente Kante, überspringen
            continue

        i = id_to_idx[src_id]
        j = id_to_idx[dst_id]

        attr_vec = _encode_edge_attr(rel)

        # gerichtete Kante i -> j
        edge_index_list.append([i, j])
        edge_attr_list.append(attr_vec)

        if undirected:
            # Rückkante j -> i hinzufügen
            edge_index_list.append([j, i])
            edge_attr_list.append(attr_vec)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        # Graph ohne Kanten – edge_index leeres Tensor-Shape [2, 0]
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_EDGE_TYPES), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Meta-Infos für spätere Auswertung/Debugging mitschleppen
    data.graph_id = g.get("graph_id")
    data.graph_label = g.get("label")
    data.graph_attrs = g.get("graph_attrs", {})

    return data


# -----------------------------
# JSONL-Helfer
# -----------------------------

def load_jsonl_graphs(path: str) -> List[Dict[str, Any]]:
    """
    Lädt eine JSONL-Datei, in der pro Zeile ein Graph-JSON steht
    (Format siehe json_graph_to_pyg).
    """
    graphs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            graphs.append(json.loads(line))
    return graphs


def load_jsonl_as_pyg(path: str, undirected: bool = True) -> List[Data]:
    """
    Komfortfunktion: läd alle Graphen aus JSONL und konvertiert sie in PyG-Data-Objekte.
    """
    graphs = load_jsonl_graphs(path)
    return [json_graph_to_pyg(g, undirected=undirected) for g in graphs]
