import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

# Interesting loss function can be found here:
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch


class RelativeMSELoss(nn.Module):
    """
    Relative loss function based on MSE loss
    """
    def __init__(self, r: float = 0.01, truncated: bool = False, threshold: float = 1e-6) -> None:
        """

        Parameters
        ----------
        r                   - residual parameter to prevent a zero divide
        truncated           - True if ReLU is activated for zero targets
        threshold           - threshold
        """
        super().__init__()
        self.r = r
        self.truncated = truncated
        self.threshold = threshold

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """

        Parameters
        ----------
        y                   - predicted output by the NN
        t                   - target output

        Returns
        -------
        loss                - loss value
        """
        if self.truncated:
            y = torch.where(t < self.threshold, F.relu(y), y)
        scale = torch.abs(t + self.r)
        y = y / scale
        t = t / scale
        loss = F.mse_loss(y, t)
        return loss


class R2Score(torch.nn.Module):
    """
    Relative loss function based on MSE loss
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor,
                t: torch.Tensor,
                adjusted: int = 0,
                multioutput: str = "uniform_average") -> torch.Tensor:  # type: ignore[override]
        sum_squared_obs, sum_obs, rss, n_obs = self._r2_score_update(y, t)
        return self._r2_score_compute(sum_squared_obs, sum_obs, rss, n_obs, adjusted, multioutput)

    def _r2_score_update(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Updates and returns variables required to compute R2 score. Checks for same shape and 1D/2D input tensors.
        Args:
            preds: Predicted tensor
            target: Ground truth tensor
        """

        self._check_same_shape(preds, target)
        if preds.ndim > 2:
            raise ValueError(
                "Expected both prediction and target to be 1D or 2D tensors,"
                f" but received tensors with dimension {preds.shape}"
            )

        sum_obs = torch.sum(target, dim=0)
        sum_squared_obs = torch.sum(target * target, dim=0)
        residual = target - preds
        rss = torch.sum(residual * residual, dim=0)
        n_obs = target.size(0)

        return sum_squared_obs, sum_obs, rss, n_obs

    def _r2_score_compute(self,
            sum_squared_obs: torch.Tensor,
            sum_obs: torch.Tensor,
            rss: torch.Tensor,
            n_obs: torch.Tensor,
            adjusted: int = 0,
            multioutput: str = "uniform_average",
    ) -> torch.Tensor:
        """Computes R2 score.
        Args:
            sum_squared_obs: Sum of square of all observations
            sum_obs: Sum of all observations
            rss: Residual sum of squares
            n_obs: Number of predictions or observations
            adjusted: number of independent regressors for calculating adjusted r2 score.
                Default 0 (standard r2 score).
            multioutput: Defines aggregation in the case of multiple output scores. Can be one
                of the following strings (default is `'uniform_average'`.):
                * `'raw_values'` returns full set of scores
                * `'uniform_average'` scores are uniformly averaged
                * `'variance_weighted'` scores are weighted by their individual variances
        Example:
            >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
            >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
            >>> sum_squared_obs, sum_obs, rss, n_obs = _r2_score_update(preds, target)
            >>> _r2_score_compute(sum_squared_obs, sum_obs, rss, n_obs, multioutput="raw_values")
            tensor([0.9654, 0.9082])
        """
        if n_obs < 2:
            raise ValueError("Needs at least two samples to calculate r2 score.")

        mean_obs = sum_obs / n_obs
        tss = sum_squared_obs - sum_obs * mean_obs
        raw_scores = 1 - (rss / tss)

        if multioutput == "raw_values":
            r2 = raw_scores
        elif multioutput == "uniform_average":
            r2 = torch.mean(raw_scores)
        elif multioutput == "variance_weighted":
            tss_sum = torch.sum(tss)
            r2 = torch.sum(tss / tss_sum * raw_scores)
        else:
            raise ValueError(
                "Argument `multioutput` must be either `raw_values`,"
                f" `uniform_average` or `variance_weighted`. Received {multioutput}."
            )

        if adjusted < 0 or not isinstance(adjusted, int):
            raise ValueError("`adjusted` parameter should be an integer larger or" " equal to 0.")

        if adjusted != 0:
            if adjusted > n_obs - 1:
                raise Warning(
                    "More independent regressions than data points in"
                    " adjusted r2 score. Falls back to standard r2 score.",
                    UserWarning,
                )
            elif adjusted == n_obs - 1:
                raise Warning("Division by zero in adjusted r2 score. Falls back to" " standard r2 score.",
                               UserWarning)
            else:
                r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - adjusted - 1)
        return -r2  # TODO: changed here to minus in order to minimize the score

    def _check_same_shape(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Check that predictions and target have the same shape, else raise error."""
        if preds.shape != target.shape:
            raise RuntimeError("Predictions and targets are expected to have the same shape")


class IOULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IOULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU
