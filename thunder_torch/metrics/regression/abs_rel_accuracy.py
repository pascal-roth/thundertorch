from typing import Any, Callable, Optional, Tuple

import torch

from thunder_torch.metrics.metric import Metric


class AbsRelAccuracy(Metric):

    def __init__(self, abs_threshold: float = 0.005, rel_threshold: float = 0.01, compute_on_step: bool = True,
                 dist_sync_on_step: bool = False, process_group: Optional[Any] = None, dist_sync_fn: Callable = None,):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct_abs", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_rel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_abs_rel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.abs_threshold = abs_threshold
        self.rel_threshold = rel_threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        rel_acc = torch.eq(target * (1-self.rel_threshold) < preds,  preds < (1+self.rel_threshold) * target)
        abs_acc = torch.eq(target - self.abs_threshold < preds,  preds < self.abs_threshold + target)

        abs_rel_acc = torch.cat((torch.t(rel_acc.reshape(1, torch.numel(rel_acc))),
                                 torch.t(abs_acc.reshape(1, torch.numel(abs_acc)))), dim=1)

        abs_rel_acc = torch.all(abs_rel_acc, dim=1)

        self.correct_rel += torch.sum(rel_acc)
        self.correct_abs += torch.sum(abs_acc)
        self.correct_abs_rel += torch.sum(abs_rel_acc)
        self.total += target.numel()

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes accuracy over state.
        """
        rel_acc = self.correct_rel.float() / self.total
        abs_acc = self.correct_abs.float() / self.total
        abs_rel_acc = self.correct_abs_rel.float() / self.total

        return abs_acc, rel_acc, abs_rel_acc
