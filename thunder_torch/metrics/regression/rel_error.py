from typing import Any, Callable, Optional, Tuple
import torch

from thunder_torch.metrics.metric import Metric


class RelError(Metric):

    def __init__(self, compute_on_step: bool = True, dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None, dist_sync_fn: Callable = None,) -> None:

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
        )

        self.add_state("rel_error", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("abs_error", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.total: torch.Tensor
        self.rel_error: torch.Tensor
        self.abs_error: torch.Tensor

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:   # type: ignore[override]
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.rel_error += torch.sum(torch.abs(((preds-target)/(target+1e-06)) * 100))
        self.abs_error += torch.sum(torch.abs(preds-target))
        self.total += target.numel()

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes accuracy over state.
        """
        mean_rel_error = self.rel_error.float() / self.total
        mean_abs_error = self.abs_error.float() / self.total
        return mean_rel_error, mean_abs_error
