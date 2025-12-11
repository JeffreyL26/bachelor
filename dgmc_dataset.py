from __future__ import annotations
import json
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from graph_pipeline import json_graph_to_pyg

#TODO: Zum Laufen bringen - Ziel vor Weihnachten

TPair = Dict[str, Any]


def load_pairs_jsonl(path: str) -> List[TPair]:
    pairs: List[TPair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


class TemplatePairDataset(Dataset):
    """
    Dataset für DGMC-Training auf Basis deiner synthetischen Template-Paare.

    Zunächst nur positive Paare (label == 1), weil dort eine echte Permutation
    (Ground Truth Matching) existiert.

    __getitem__ liefert EIN torch_geometric.data.Data-Objekt mit Feldern:
    - x_s, edge_index_s, edge_attr_s  (Source-Graph)
    - x_t, edge_index_t, edge_attr_t  (Target-Graph)
    - y  (Permutation: index in source -> index in target)
    """

    def __init__(self, pairs_path: str, use_only_positive: bool = True, undirected: bool = True):
        all_pairs = load_pairs_jsonl(pairs_path)
        if use_only_positive:
            self.pairs = [p for p in all_pairs if p.get("label") == 1]
        else:
            self.pairs = all_pairs
        self.undirected = undirected

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Data:
        p = self.pairs[idx]

        # 1) JSON -> PyG-Graphen
        g_a = json_graph_to_pyg(p["graph_a"], undirected=self.undirected)
        g_b = json_graph_to_pyg(p["graph_b"], undirected=self.undirected)

        # In unseren synthetischen positiven Paaren gilt: gleiche Node-Anzahl
        assert g_a.x.size(0) == g_b.x.size(0), "Positive Paare müssen gleiche Knotenzahl haben"

        # 2) Ground Truth Permutation
        perm = p.get("perm")
        if perm is None:
            raise RuntimeError("Diese Dataset-Konfiguration erwartet nur positive Paare mit 'perm'.")

        y = torch.tensor(perm, dtype=torch.long)  # Shape [num_nodes]

        # 3) Kombiniertes Data-Objekt bauen
        # Wichtig: num_nodes setzen, damit PyG nicht meckert.
        num_nodes = g_a.x.size(0)

        data = Data(
            x_s=g_a.x,
            edge_index_s=g_a.edge_index,
            edge_attr_s=g_a.edge_attr,

            x_t=g_b.x,
            edge_index_t=g_b.edge_index,
            edge_attr_t=g_b.edge_attr,

            y=y,
            num_nodes=num_nodes,   # <- Fix für die 'num_nodes'-Warnung
        )

        return data