from typing import Any, Callable, Optional, Tuple

import torch

from thunder_torch.metrics.metric import Metric


class RelIntervals(Metric):

    def __init__(self, rel_threshold: float = 0.01, compute_on_step: bool = True,
                 dist_sync_on_step: bool = False, process_group: Optional[Any] = None, dist_sync_fn: Callable = None,):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct_rel_left", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_rel_right", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_rel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.rel_threshold = rel_threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        rel_acc = torch.eq(target * (1-self.rel_threshold) < preds,  preds < (1+self.rel_threshold) * target)
        rel_acc_left = torch.eq(target * (1-self.rel_threshold) < preds,  preds < target)
        rel_acc_right = torch.eq(target < preds,  preds < (1+self.rel_threshold) * target)

        self.correct_rel += torch.sum(rel_acc)
        self.correct_rel_left += torch.sum(rel_acc_left)
        self.correct_rel_right += torch.sum(rel_acc_right)
        self.total += target.numel()

    def compute(self) -> Tuple[float, float, float]:
        """
        Computes accuracy over state.
        """
        rel_acc = self.correct_rel.float() / self.total
        rel_acc_left = self.correct_rel_left.float() / self.total
        rel_acc_right = self.correct_rel_right.float() / self.total

        return rel_acc_left, rel_acc_right, rel_acc
