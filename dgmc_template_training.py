from __future__ import annotations
import os
import random
import torch
from torch.optim import Adam
from dgmc import DGMC
from dgmc.models import GIN
from dgmc_dataset import TemplatePairDataset

#TODO: Kommentare und Dokumentation revamp

def generate_y(y_col: torch.Tensor, device: torch.device) -> torch.LongTensor:
    """
    Wandelt einen Vektor [num_nodes] in ein Matching-Array [2, num_nodes] um:
    y[0, k] = Index im Source-Graph,
    y[1, k] = Index im Target-Graph.
    """
    y_col = y_col.to(device)                   # z.B. [0, 2, 1, ...]
    y_row = torch.arange(y_col.size(0), device=device)  # [0, 1, 2, ...]
    return torch.stack([y_row, y_col], dim=0)           # Shape [2, num_nodes]


def train_epoch(model: DGMC,
                dataset: TemplatePairDataset,
                optimizer: Adam,
                device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    # einfache Shuffle-Reihenfolge ohne DataLoader
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        data = dataset[idx]
        data = data.to(device)

        optimizer.zero_grad()

        # Single-Pair-Forward, wie im test() der DGMC-Beispiele:
        # Batch-Argumente sind None, weil wir nicht batchen.
        S0, SL = model(
            data.x_s, data.edge_index_s, data.edge_attr_s, None,
            data.x_t, data.edge_index_t, data.edge_attr_t, None,
        )

        # Ground Truth Matching aufbauen
        y = generate_y(data.y, device)

        # Loss (Initial + verfeinertes Matching)
        loss = model.loss(S0, y)
        if model.num_steps > 0:
            loss = loss + model.loss(SL, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(dataset), 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = os.path.dirname(os.path.abspath(__file__))
    pairs_path = os.path.join(base, "data", "synthetic_training_pairs.jsonl")

    dataset = TemplatePairDataset(pairs_path, use_only_positive=True, undirected=True)

    # Feature-Dimension aus erstem Datapoint
    sample = dataset[0]
    in_channels = sample.x_s.size(-1)
    hidden_dim = 64

    # psi_1 & psi_2: einfache GIN-Encoder
    psi_1 = GIN(in_channels, hidden_dim, num_layers=3)
    psi_2 = GIN(in_channels, hidden_dim, num_layers=3)

    model = DGMC(psi_1, psi_2, num_steps=10, k=-1).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, dataset, optimizer, device)
        print(f"Epoch {epoch:02d} | loss = {loss:.4f}")

    # Modell speichern
    out_path = os.path.join(base, "data", "dgmc_templates.pt")
    torch.save(model.state_dict(), out_path)
    print("DGMC-Template-Modell gespeichert unter:", out_path)


if __name__ == "__main__":
    main()