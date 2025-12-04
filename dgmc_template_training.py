from __future__ import annotations
import os
import random
import torch
from torch.optim import Adam
from dgmc import DGMC
from dgmc.models import GIN
from dgmc_dataset import TemplatePairDataset

#TODO: Zum Laufen bringen - Ziel vor Weihnachten

def generate_y(y_col: torch.Tensor, device: torch.device) -> torch.LongTensor:
    """
    Wandelt einen Vektor [num_nodes] in ein Matching-Array [2, num_nodes] um:
    y[0, k] = Index im Source-Graph,
    y[1, k] = Index im Target-Graph.
    """
    y_col = y_col.to(device)                   # z.B. [0, 2, 1, ...]
    y_row = torch.arange(y_col.size(0), device=device)  # [0, 1, 2, ...]
    return torch.stack([y_row, y_col], dim=0)           # Shape [2, num_nodes]



#if __name__ == "__main__":
    #main()