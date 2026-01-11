from __future__ import annotations

"""
dgmc_split_metrics.py

Small helper module to compute additional, thesis-friendly metrics during DGMC training
WITHOUT changing DGMC itself.

Motivation (Goal 2/3 reporting):
- Separate "full source supervision" pairs (|y| == |V_s|) from "partial supervision"
  pairs (|y| < |V_s|) inside each batch/epoch.
- Report loss/accuracy separately for these two regimes to make DGMC limitations visible,
  without claiming "bad dataset".

Assumptions (consistent with your current y-based dataset/collate):
- DGMC returns dense correspondence matrices S0 / SL in the common setting k=-1.
- y is a LongTensor [2, K] with:
    y[0] = global source row indices over the flattened source batch
    y[1] = *local* target indices within each target graph (no target offset!)
- batch_s is a LongTensor [Ns_total] mapping each source node to its pair index in the batch.

If you switch to sparse DGMC (k >= 1) and DGMC returns a sparse structure, this module
will attempt to convert to dense via `to_dense()` for metric computation. For your small graphs
this is fine; for large graphs you should keep k=-1 when using these metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

EPS = 1e-15


def _to_dense(S: Any) -> torch.Tensor:
    """Convert DGMC output S to a dense torch.Tensor if needed."""
    if isinstance(S, torch.Tensor):
        return S
    # Many sparse types (torch_sparse.SparseTensor) implement to_dense()
    if hasattr(S, "to_dense"):
        Sd = S.to_dense()
        if not isinstance(Sd, torch.Tensor):
            raise TypeError(f"to_dense() did not return torch.Tensor (got {type(Sd)})")
        return Sd
    raise TypeError(f"Unsupported S type for metric computation: {type(S)}")


@dataclass
class SplitAgg:
    # sums over supervised matches (y columns)
    loss0_sum: float = 0.0
    lossL_sum: float = 0.0
    correct_sum: float = 0.0
    matches: int = 0
    pairs: int = 0


def _loss_and_correct(S: Any, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      loss_vec: [K]  negative log prob on the supervised correspondences
      correct_vec: [K]  1.0 if argmax row equals the supervised col
    """
    Sd = _to_dense(S)
    rows = y[0]
    cols = y[1]
    prob = Sd[rows, cols].clamp(min=EPS)
    loss_vec = -prob.log()

    pred = Sd.argmax(dim=-1)
    correct_vec = (pred[rows] == cols).float()
    return loss_vec, correct_vec


def _full_partial_masks(batch_s: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      full_match_mask: [K] True if the match belongs to a pair with full source coverage (K_i == ns_i)
      partial_match_mask: [K] inverse
      full_pair_mask: [B] True for pairs with full source coverage
    """
    if batch_s.numel() == 0:
        # no nodes
        K = int(y.size(1))
        full_match_mask = torch.zeros((K,), dtype=torch.bool, device=y.device)
        partial_match_mask = ~full_match_mask
        full_pair_mask = torch.zeros((0,), dtype=torch.bool, device=y.device)
        return full_match_mask, partial_match_mask, full_pair_mask

    graph_idx = batch_s[y[0]]  # [K]
    B = int(batch_s.max().item()) + 1

    ns_per_pair = torch.bincount(batch_s, minlength=B)           # [B]
    k_per_pair = torch.bincount(graph_idx, minlength=B)          # [B]
    full_pair_mask = (k_per_pair == ns_per_pair)                 # [B]

    full_match_mask = full_pair_mask[graph_idx]                  # [K]
    partial_match_mask = ~full_match_mask
    return full_match_mask, partial_match_mask, full_pair_mask


class EpochSplitMetrics:
    """
    Aggregate DGMC metrics per epoch, split into:
      - overall: all supervised matches
      - full: matches belonging to pairs with |y| == |V_s|
      - partial: matches belonging to pairs with |y| < |V_s|
    """
    def __init__(self, num_steps: int) -> None:
        self.num_steps = int(num_steps)
        self.overall = SplitAgg()
        self.full = SplitAgg()
        self.partial = SplitAgg()

    def update(self, S0: Any, SL: Any, y: torch.Tensor, batch_s: torch.Tensor) -> None:
        """
        Update aggregations from one batch.
        """
        # Use final matrix for "L" metrics:
        S_final = SL if self.num_steps > 0 else S0

        loss0_vec, _ = _loss_and_correct(S0, y)
        lossL_vec, corr_vec = _loss_and_correct(S_final, y)

        full_match, partial_match, full_pair_mask = _full_partial_masks(batch_s, y)

        K = int(y.size(1))
        B = int(batch_s.max().item()) + 1 if batch_s.numel() else 0
        full_pairs = int(full_pair_mask.sum().item())
        partial_pairs = int(B - full_pairs)

        # overall
        self.overall.loss0_sum += float(loss0_vec.sum().item())
        self.overall.lossL_sum += float(lossL_vec.sum().item())
        self.overall.correct_sum += float(corr_vec.sum().item())
        self.overall.matches += K
        self.overall.pairs += B

        # full
        if full_pairs > 0:
            self.full.loss0_sum += float(loss0_vec[full_match].sum().item())
            self.full.lossL_sum += float(lossL_vec[full_match].sum().item())
            self.full.correct_sum += float(corr_vec[full_match].sum().item())
            self.full.matches += int(full_match.sum().item())
            self.full.pairs += full_pairs

        # partial
        if partial_pairs > 0:
            self.partial.loss0_sum += float(loss0_vec[partial_match].sum().item())
            self.partial.lossL_sum += float(lossL_vec[partial_match].sum().item())
            self.partial.correct_sum += float(corr_vec[partial_match].sum().item())
            self.partial.matches += int(partial_match.sum().item())
            self.partial.pairs += partial_pairs

    @staticmethod
    def _finalize_one(agg: SplitAgg) -> Dict[str, Any]:
        if agg.matches <= 0:
            return {
                "loss_total": float("nan"),
                "loss_s0": float("nan"),
                "loss_sl": float("nan"),
                "acc": float("nan"),
                "matches": 0,
                "pairs": int(agg.pairs),
            }
        loss_s0 = agg.loss0_sum / agg.matches
        loss_sl = agg.lossL_sum / agg.matches
        return {
            "loss_total": (agg.loss0_sum + agg.lossL_sum) / agg.matches,
            "loss_s0": loss_s0,
            "loss_sl": loss_sl,
            "acc": agg.correct_sum / agg.matches,
            "matches": int(agg.matches),
            "pairs": int(agg.pairs),
        }

    def finalize(self) -> Dict[str, Dict[str, Any]]:
        return {
            "overall": self._finalize_one(self.overall),
            "full": self._finalize_one(self.full),
            "partial": self._finalize_one(self.partial),
        }
