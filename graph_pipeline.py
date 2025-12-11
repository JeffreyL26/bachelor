from __future__ import annotations
import json
from typing import Dict, Any, List, Optional, Iterable
import torch
from torch_geometric.data import Data

# ------------------------------
# FEATURE - VOKABULAR
# ------------------------------
"""
JSON-Graph hat die Form:
{
  "nodes": [
    {"id": "12345678", "type": "MaLo", "attrs": {"direction": "consumption"}},
    {"id": "DE00...", "type": "MeLo", "attrs": {"voltage_level": "E06"}},
    {"id": "1ABC...", "type": "TR", "attrs": {}}
  ],
  "edges": [
    {"src": "DE00...", "dst": "1ABC...", "rel": "METR"},
    {"src": "DE00...", "dst": "12345678", "rel": "MEMA"}
  ]
}

GNN kann mit solchen Strings nicht anfangen, also übersetzen wirr in Tensor mit Features
"""
# Knotentypen One-Hot-Encoding
NODE_TYPES = {
    "MaLo": 0,
    "MeLo": 1,
    "TR":   2,
    "NeLo": 3,
}
NUM_NODE_TYPES = len(NODE_TYPES)

# MaLo-Richtung
DIRECTIONS = {
    "consumption": 0,
    "generation":  1,
    # alles andere → unknown
}
DIR_UNKNOWN_INDEX = len(DIRECTIONS)
NUM_DIRECTIONS = len(DIRECTIONS) + 1  # + unknown

# MeLo-Funktion für Templates
MELO_FUNCTIONS = {
    "N": 0,
    "H": 1,
    "D": 2,
    "S": 3,
    # → unknown
}
MELO_FUNC_UNKNOWN_INDEX = len(MELO_FUNCTIONS)
NUM_MELO_FUNCTIONS = len(MELO_FUNCTIONS) + 1

# Spannungsebene
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


# ------------------------------
# ONE-HOT-ENCODINGS
# ------------------------------

def _one_hot(index: int, length: int) -> List[float]:
    """
    Erzeugt eine One-Hot-Kodierung.
    :param index: An dieser Stelle steht 1.0 im Vektor, ansonsten 0.0
    :param length: Länge der Kodierung
    :return: One-Hot-Vektor mit float, das PyG typischerweise float nutzt
    """
    v = [0.0] * length
    if 0 <= index < length:
        v[index] = 1.0
    return v


def _encode_node_features(node: Dict[str, Any]) -> List[float]:
    """
    Kodiert einen JSON-Knoten in einen Feature-Vektor.

    Erwartetes Node-Format:
    {
        "id": "<string>",
        "type": "MaLo" | "MeLo" | "TR" | "NeLo",
        "attrs": { ... }
    }
    :param node: Knoten-Dict
    """
    ntype = node.get("type")                                        # Knotentyp
    attrs = node.get("attrs", {}) or {}                             # Attributdict

    # Knotentyp zu One-Hot
    type_idx = NODE_TYPES.get(ntype, None)
    if type_idx is None:
        # Unbekannter Typ bedeutet alles 0, optional erweitern
        type_vec = [0.0] * NUM_NODE_TYPES
    else:
        type_vec = _one_hot(type_idx, NUM_NODE_TYPES)

    # MaLo-Richtung
    raw_dir = attrs.get("direction")
    dir_idx = DIRECTIONS.get(raw_dir, DIR_UNKNOWN_INDEX)            # Bei MeLo, TR, NeLo ist das None
    dir_vec = _one_hot(dir_idx, NUM_DIRECTIONS)

    # MeLo-Funktion
    raw_fn = attrs.get("function") if ntype == "MeLo" else None
    fn_idx = MELO_FUNCTIONS.get(raw_fn, MELO_FUNC_UNKNOWN_INDEX)
    fn_vec = _one_hot(fn_idx, NUM_MELO_FUNCTIONS)

    # Spannungsebene
    raw_volt = attrs.get("voltage_level") if ntype == "MeLo" else None
    volt_idx = VOLTAGE_LEVELS.get(raw_volt, VOLT_UNKNOWN_INDEX)
    volt_vec = _one_hot(volt_idx, NUM_VOLTAGE_LEVELS)

    # LBS_OBJECT_LEVEL für Template-Graphen, einfache Zahl (bei Ist-Konstrukte default 0.0)
    level = float(attrs.get("level", 0.0))

    # Konkatenation aller Feature-Blöcke: 4 Knotentypen + 3 MaLo-Richtungen + 5 MeLo-Funktionen + 3 Spannungsebenen + 1 Level = 16
    return type_vec + dir_vec + fn_vec + volt_vec + [level]


def _encode_edge_attr(rel: str) -> List[float]:
    """
    Kodiert Kanten-Typ als One-Hot-Feature.
    "MEMA", "METR", "MENE", "MEME".
    :param rel: Kanten-Typ
    """
    idx = EDGE_TYPES.get(rel, EDGE_UNKNOWN_INDEX)
    return _one_hot(idx, NUM_EDGE_TYPES)


# ------------------------------
# JSON-GRAPH ZU PyG DATA
# ------------------------------

def json_graph_to_pyg(
    g: Dict[str, Any],
    undirected: bool = True,                                        # Damit das GNN in beide Richtungen Nachrichten senden kann
) -> Data:
    """
    Konvertiert einen Graphen im JSON-Format (ist_graphs_small.jsonl
    oder lbs_templates.jsonl) in ein torch_geometric.data.Data-Objekt

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

    Erwartete Rückgabe:
    Data(
        x=[num_nodes, 16],
        edge_index=[2, num_edges],
        edge_attr=[num_edges, 5],
        graph_id=...,
        graph_attrs=...
    )
    """

    # Knoten und Kanten aus dict
    nodes = g.get("nodes", [])                                      # Bsp.: {"id": "50414974078", "type": "MaLo", "attrs": {"direction": "consumption"}}
    edges = g.get("edges", [])                                      # Bsp.: {"src": "DE0071...", "dst": "50414974078", "rel": "MEMA"}

    # Knoten-ID zu Index im Node-Array mappen
    # Beispiel:
    # nodes[0]["id"] = "123456789"     #MaLo
    # nodes[1]["id"] = "DE00..."       #MeLo
    # nodes[2]["id"] = "JEFLEHMA..."   #TR
    id_to_idx: Dict[str, int] = {n["id"]: i for i, n in enumerate(nodes)}

    # Node-Features x = [num_nodes, 16]
    # Liste von Listen
    # Beispiel:
    # x_list = [
    #     [... 16 floats für Node 0...],
    #     [... 16 floats für Node 1...],
    #     [... 16 floats für Node 2...],]
    x_list: List[List[float]] = [_encode_node_features(n) for n in nodes]

    # Daraus ein Tensor → Knoten-Feature-Matrix
    x = torch.tensor(x_list, dtype=torch.float)

    # Kanten und deren Features
    edge_index_list: List[List[int]] = []                           # Liste von Quell- und Zielknotenpaaren
    edge_attr_list: List[List[float]] = []                          # Liste der dazugehörigen Kanten-Features

    for e in edges:
        # Je Quell-, Zielknoten und Relation extrahierem
        src_id = e.get("src")
        dst_id = e.get("dst")
        rel = e.get("rel")

        if src_id not in id_to_idx or dst_id not in id_to_idx:
            # Inkonsistente Kante, überspringen
            continue

        # Indizes  Quell- und Zielknoten
        i = id_to_idx[src_id]
        j = id_to_idx[dst_id]

        # Kantentyp Featurevektor
        attr_vec = _encode_edge_attr(rel)

        # gerichtete Kante i → j
        edge_index_list.append([i, j])
        edge_attr_list.append(attr_vec)

        if undirected:
            # Rückkante j → i hinzufügen
            edge_index_list.append([j, i])
            edge_attr_list.append(attr_vec)

        # Bsiepiel:
        # edge_index_list: [[MeLo Index, MaLo Index], [MaLo Index, MeLo Index]]
        # edge_attr_list: [attr_vec, attr_vec]

    # Umwandlung in PyG-Torch-Tensoren
    # Es gibt Kanten
    if edge_index_list:
        # Transponieren, den PyG hat Konvention [2 Spalten (src, dst), Anzahl Kanten], ich habe es ursprünglich andersherum. contiguous(), um aktuelle Reihenfolge beizubehalten
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    # Es gibt keine Kanten
    else:
        # edge_index leeres Tensor-Shape [2, 0]
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_EDGE_TYPES), dtype=torch.float)

    # Standardformat für DGMC (theoretisch auch andere PyG-Modelle: MERKEN!)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Meta-Infos für Auswertung/Debugging
    data.graph_id = g.get("graph_id")
    data.graph_label = g.get("label")
    data.graph_attrs = g.get("graph_attrs", {})

    return data


# ------------------------------
# JSONL
# ------------------------------

def load_jsonl_graphs(path: str) -> List[Dict[str, Any]]:
    """
    Lädt eine JSONL-Datei, in der pro Zeile ein JSON-Graph steht
    Format @json_graph_to_pyg
    :param path: Pfad zur JSONL-Datei
    :return: Liste von Graphen
    """

    # Liste mit Graphen
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
    Lädt alle Graphen aus JSONL und konvertiert sie in PyG-Data-Objekte.
    :param path: Pfad zur JSONL-Datei
    :param undirected: Graphen gerichtet (=true) oder ungerichtet (= false)
    :return: Liste von PyG-Data-Objekten
    """
    graphs = load_jsonl_graphs(path)
    return [json_graph_to_pyg(g, undirected=undirected) for g in graphs]
