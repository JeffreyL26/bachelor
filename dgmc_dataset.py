from __future__ import annotations
import json
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from graph_pipeline import json_graph_to_pyg

# Synthetische Paare aus @synthetic_pair_builder.generate_pairs nehmen und daraus PyTorch-Datasets bauen

TPair = Dict[str, Any]

# ------------------------------
# SYNTHETISCHE PAARE LADEN
# ------------------------------

def load_pairs_jsonl(path: str) -> List[TPair]:
    """
    Paare aus JSONL laden
    :param path: Pfad zur JSONL-Datei
    :return: Liste von Paaren
    """
    pairs: List[TPair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs

# ------------------------------
# TRAININGS-DATASET
# ------------------------------

class TemplatePairDataset(Dataset):
    """
    Dataset für DGMC-Training auf Basis von synthetischen Template-Paaren

    Zunächst nur positive Paare (label = 1), weil dort eine echte Permutation
    (Ground-Truth-Matching) existiert.

    Paar von Graphen mit Ground-Truth-Matching anstelle von ein Graph pro Sample
    """

    def __init__(self, pairs_path: str, use_only_positive: bool = True, undirected: bool = True):
        """
        Datei laden und Paare filtern
        :param pairs_path: Pfad zur JSONL-Datei der synthetischen Paare
        :param use_only_positive: Nur positive Paare verwenden (label = 1)
        :param undirected: Ungerichtet (= true) oder gerichtet (= false)
        """
        all_pairs = load_pairs_jsonl(pairs_path)
        if use_only_positive:
            self.pairs = [p for p in all_pairs if p.get("label") == 1]
        else:
            self.pairs = all_pairs
        self.undirected = undirected

    def __len__(self) -> int:
        """
        Gibt Dataset-Größe zurück
        :return: Dataset-Größe
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Data:
        """
        Liefert 1 torch_geometric.data.Data-Objekt mit Feldern:
            - x_s, edge_index_s, edge_attr_s: (Source-Graph)
            - x_t, edge_index_t, edge_attr_t: (Ziel-Graph)
            - y  (Permutation: index in source -> index in target)
        :param idx: Index für Paar aus der Liste
        :return: PyG-Data-Objekt
        """
        p = self.pairs[idx]

        # JSON zu PyG mit Pipeline-Funktion
        g_a = json_graph_to_pyg(p["graph_a"], undirected=self.undirected)
        g_b = json_graph_to_pyg(p["graph_b"], undirected=self.undirected)

        # Test positive Paare: gleiche Node-Anzahl
        assert g_a.x.size(0) == g_b.x.size(0), "Positive Paare müssen gleiche Knotenzahl haben."

        # Ground-Truth-Permutation extrahieren
        perm = p.get("perm")
        if perm is None:
            raise RuntimeError("Dataset-Konfiguration erwartet nur positive Paare mit 'perm'.")

        # LongTensor der Länge num_nodes als "Zuordnungsfunktion"
        y = torch.tensor(perm, dtype=torch.long)  # Shape [num_nodes]

        # Data-Objekt bauen, das zwei Graphen beinhaltet
        # num_nodes setzen, damit PyG nicht rummault
        num_nodes = g_a.x.size(0)

        # Direkter Input für DGMC
        data = Data(
            x_s=g_a.x,                                                      # Quellgraph - Knoten-Feature-Matrix
            edge_index_s=g_a.edge_index,                                    # Liste von Quell- und Zielknotenpaaren
            edge_attr_s=g_a.edge_attr,                                      # Liste der dazugehörigen Kanten-Features

            x_t=g_b.x,                                                      # Zielgraph - Knoten-Feature-Matrix
            edge_index_t=g_b.edge_index,                                    # Liste von Quell- und Zielknotenpaaren
            edge_attr_t=g_b.edge_attr,                                      # Liste der dazugehörigen Kanten-Features

            y=y,                                                            # Ground-Truth-Permutation
            num_nodes=num_nodes,                                            # Fix für die num_nodes-Warnung
        )

        return data