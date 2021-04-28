import torch
import torch.nn as nn
import torch.nn.functional as F


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
