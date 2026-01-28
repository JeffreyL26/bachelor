"""
dgmc_dataset.py
PyG/DGMC dataset helper für synthetische Trainingpaare
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch, Data

# Project import: converts JSON graph -> PyG Data(x, edge_index, edge_attr, ...)
import graph_pipeline as gp

#Typisierter Graph und Paar
TGraph = Dict[str, Any]
TPair = Dict[str, Any]


@dataclass
class PairItem:
    """
    Ein Trainings- bzw. Evaluierungspaar
    """

    source_data: Data
    target_data: Data
    y_srctgt_pair: Optional[torch.Tensor]
    label: int
    meta: Dict[str, Any]


def _parse_y(y: Any) -> Optional[torch.Tensor]:
    """
    y parsen in einen Tensor [2, K]
    y = [[src_indices...], [tgt_indices...]]
    """
    if y is None:
        return None

    if isinstance(y, torch.Tensor):
        if y.dtype != torch.long:
            y = y.long()
        if y.dim() == 2 and y.size(0) == 2:
            return y
        return None

    if not isinstance(y, (list, tuple)) or len(y) != 2:
        return None

    source_indizes, target_indizes = y
    if not isinstance(source_indizes, (list, tuple)) or not isinstance(target_indizes, (list, tuple)):
        return None
    if len(source_indizes) != len(target_indizes):
        return None
    if len(source_indizes) == 0:
        # Leere Korrespondenzen (fallback, damit wir auch validity aus dem Proposal abdecken)
        return torch.empty((2, 0), dtype=torch.long)

    try:
        source = torch.tensor([int(x) for x in source_indizes], dtype=torch.long)
        target = torch.tensor([int(x) for x in target_indizes], dtype=torch.long)
    except Exception:
        return None

    return torch.stack([source, target], dim=0)


def _tensor_abnehmprogramm(data: Data) -> Data:
    """
    Nur Tensor-Felder behalten, die DGMC auch verwerten kann (keine graph_attrs, node_ids)
    """
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return out


def _y_validierer(
    y: torch.Tensor,
    node_amount_source: int,
    node_amount_target: int,
    eins_zu_eins_korrespondenz: bool = True,
    jede_source_coverage: bool = False,
) -> bool:
    """
    Validiert, ob y richtig ankommt (also Indizes im richtigen Bereich und optimalerweise 1-1 (kontrollgruppe mit Permutationstraining))
    """
    #DImensionen
    if y.dim() != 2 or y.size(0) != 2:
        return False
    if y.numel() == 0:
        return True

    #Inhalt überprüfen
    source = y[0]
    target = y[1]
    if source.min().item() < 0 or source.max().item() >= node_amount_source:
        return False
    if target.min().item() < 0 or target.max().item() >= node_amount_target:
        return False

    if eins_zu_eins_korrespondenz:
        #Keine Duplikate
        if source.unique().numel() != source.numel():
            return False
        if target.unique().numel() != target.numel():
            return False

    if jede_source_coverage:
        #Alle Source-Knoten müssen matched sein
        if source.numel() != node_amount_source:
            return False

    return True


def _maybe_swap_source_target(
    source_data: Data,
    target_data: Data,
    y_srctgt_pair: Optional[torch.Tensor],
    kleinere_source: bool,
) -> Tuple[Data, Data, Optional[torch.Tensor], bool]:
    """
    DGMC kann auch injektiv, also optional (A,B) tauschen, dass |source| <= |target|
    :return: (source_data, target_data, y_srctgt_pair_swapped, did_swap).
    """
    if not kleinere_source:
        return source_data, target_data, y_srctgt_pair, False

    source_node_number = int(source_data.num_nodes)
    target_node_number = int(target_data.num_nodes)
    if source_node_number <= target_node_number:
        return source_data, target_data, y_srctgt_pair, False

    #Tausch
    if y_srctgt_pair is None:
        y_srctgt_pair_swapped = None
    else:
        y_srctgt_pair_swapped = torch.stack([y_srctgt_pair[1], y_srctgt_pair[0]], dim=0)
    return target_data, source_data, y_srctgt_pair_swapped, True


class DGMCPairJsonlDataset(torch.utils.data.Dataset):
    """
    Lädt JSONL aus synthetischen Trainingspaaren
    """

    def __init__(
        self,
        pairs_path: str,
        *,
        undirected: bool = True,
        use_only_positive: bool = True,
        allow_pairs_without_y: bool = False,
        prefer_source_smaller: bool = True,
        strip_meta_from_data: bool = True,
        require_one_to_one: bool = True,
        require_full_source_coverage: bool = False,
    ) -> None:
        super().__init__()

        self.pairs_path = pairs_path
        self.undirected = undirected
        self.use_only_positive = use_only_positive
        self.allow_pairs_without_y = allow_pairs_without_y
        self.prefer_source_smaller = prefer_source_smaller
        self.strip_meta_from_data = strip_meta_from_data
        self.require_one_to_one = require_one_to_one
        self.require_full_source_coverage = require_full_source_coverage

        if not os.path.exists(pairs_path):
            raise FileNotFoundError(pairs_path)

        self._pairs: List[TPair] = []
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSON on line {line_no} of {pairs_path}: {e}") from e

                label = int(p.get("label", 1))
                if self.use_only_positive and label != 1:
                    continue

                has_y = p.get("y") is not None
                if (not has_y) and (not self.allow_pairs_without_y):
                    #Paare ohne y nutzen uns erstmal nichts
                    continue

                if "graph_a" not in p or "graph_b" not in p:
                    raise ValueError(
                        f"Pair on line {line_no} misses 'graph_a'/'graph_b' keys. "
                        "Expected synthetic_pair_builder.py output."
                    )

                self._pairs.append(p)

        if not self._pairs:
            raise ValueError(
                "No usable pairs were loaded. "
                "If JSONL contains negatives (label=0) or y=None, "
                "set allow_pairs_without_y=True or use_only_positive=False."
            )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> PairItem:
        p = self._pairs[idx]

        source_graph: TGraph = p["graph_a"]
        target_graph: TGraph = p["graph_b"]
        label = int(p.get("label", 1))
        y_srctgt_pair = _parse_y(p.get("y"))

        #JSON in verwendbare Daten umschreiben
        source_data = gp.json_graph_to_pyg(source_graph, undirected=self.undirected)
        target_data = gp.json_graph_to_pyg(target_graph, undirected=self.undirected)

        if self.strip_meta_from_data:
            source_data = _tensor_abnehmprogramm(source_data)
            target_data = _tensor_abnehmprogramm(target_data)

        #Optionaler Tausch
        data_s, data_t, y_srctgt_pair, did_swap = _maybe_swap_source_target(
            source_data, target_data, y_srctgt_pair, kleinere_source=self.prefer_source_smaller
        )

        # y validieren
        if y_srctgt_pair is not None:
            passt = _y_validierer(
                y_srctgt_pair,
                node_amount_source=int(data_s.num_nodes),
                node_amount_target=int(data_t.num_nodes),
                eins_zu_eins_korrespondenz=self.require_one_to_one,
                jede_source_coverage=self.require_full_source_coverage,
            )
            if not passt:
                raise ValueError(
                    f"Invalid y on idx={idx} (swap={did_swap}): "
                    f"|V_s|={int(data_s.num_nodes)}, |V_t|={int(data_t.num_nodes)}, y.shape={tuple(y_srctgt_pair.shape)}"
                )

        meta: Dict[str, Any] = {
            "idx": idx,
            "did_swap": did_swap,
            "label": label,
            "graph_id_a": (source_graph.get("graph_id") if isinstance(source_graph, dict) else None),
            "graph_id_b": (target_graph.get("graph_id") if isinstance(target_graph, dict) else None),
        }
        if "aug" in p:
            meta["aug"] = p.get("aug")

        return PairItem(source_data=data_s, target_data=data_t, y_srctgt_pair=y_srctgt_pair, label=label, meta=meta)

    def describe(self, max_items: int = 3) -> str:
        """
        Kurze Zusammenfassung
        """
        n = len(self)
        items = [self[i] for i in range(min(max_items, n))]
        teile = [f"DGMCPairJsonlDataset(n={n}, undirected={self.undirected}, prefer_source_smaller={self.prefer_source_smaller})"]
        for it in items:
            k = int(it.y_srctgt_pair.size(1)) if it.y_srctgt_pair is not None else 0
            teile.append(
                f"  - idx={it.meta.get('idx')} |V_s| (Anzahl Source Nodes) ={int(it.source_data.num_nodes)} |V_t| (Anzahl Target Nodes) ={int(it.target_data.num_nodes)} |y| (Anzahl Ground-Truth-Knotenpaare) ={k} swap={it.meta.get('did_swap')}"
            )
        return "\n".join(teile)


def pair_batcher(batch: Sequence[PairItem]) -> Dict[str, Any]:
    """
    Erstellt aus einer Liste von Graph-Paaren einen PyG-Batch für Source und Target und wandelt die
    lokalen Korrespondenzen y_srctgt_pair in das von DGMC erwartete Batch-Format um (Source-Indizes global mit Offset,
    da wir nicht immer jeden Graphen bei 0 anfangen zu nummerieren. Target-Indizes lokal pro Graph).
    """
    if not batch:
        raise ValueError("Empty batch")

    #Graphen batchen
    batch_s = Batch.from_data_list([it.source_data for it in batch])
    batch_t = Batch.from_data_list([it.target_data for it in batch])

    #DGMC y bauen mit globalen source Zeilen und lokalen target Spalten
    y_zeilen: List[int] = []
    y_spalten: List[int] = []
    src_offset = 0
    for it in batch:
        ns = int(it.source_data.num_nodes)
        nt = int(it.target_data.num_nodes)

        if it.y_srctgt_pair is None:
            raise ValueError(
                "pair_batcher encountered an item with y_srctgt_pair=None "
                "Filter such pairs (allow_pairs_without_y=False)"
            )

        y_lokal = it.y_srctgt_pair
        if y_lokal.numel() == 0:
            src_offset += ns
            continue

        #Lokales y[0] muss jetzt mit offset verschoben werden (s. Docstring)
        source = (y_lokal[0] + src_offset).tolist()
        target = y_lokal[1].tolist()  # Unverändert lassen, da Target-Indizes pro Target-Graph lokal erwartet werden und nicht global über alle Target-Graphen

        # Grenzen-Check
        if max(target, default=-1) >= nt:
            raise ValueError(f"Target index out of bounds in batch item idx={it.meta.get('idx')}: max_tgt={max(target)} nt={nt}")
        if max(source, default=-1) >= (src_offset + ns):
            raise ValueError(f"Source index out of bounds after offset in batch item idx={it.meta.get('idx')}")

        y_zeilen.extend(source)
        y_spalten.extend(target)
        src_offset += ns

    y = torch.tensor([y_zeilen, y_spalten], dtype=torch.long)

    labels = torch.tensor([int(it.label) for it in batch], dtype=torch.long)
    meta = [it.meta for it in batch]
    return {"batch_s": batch_s, "batch_t": batch_t, "y": y, "label": labels, "meta": meta}


if __name__ == "__main__":
    #Tester
    import random
    from torch.utils.data import DataLoader

    random.seed(42)

    base = os.path.dirname(os.path.abspath(__file__))
    pairs_path = os.path.join(base, "data", "synthetic_training_pairs.jsonl")
    #pairs_path = os.path.join(base, "data", "synthetic_training_pairs_control.jsonl")
    ds = DGMCPairJsonlDataset(pairs_path, use_only_positive=True)
    print(ds.describe())

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=pair_batcher)
    b = next(iter(dl))
    print("batch_s:", b["batch_s"].num_graphs, "graphs |", b["batch_s"].num_nodes, "nodes")
    print("batch_t:", b["batch_t"].num_graphs, "graphs |", b["batch_t"].num_nodes, "nodes")
    print("y:", tuple(b["y"].shape), "| labels:", b["label"].tolist())
