import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class RelativeMSELoss(pl.LightningModule):
    """
    Relative loss function based on MSE loss
    """
    def __init__(self, r=0.01, truncated=False, threshold=1e-6) -> None:
        """
        :param r: residual parameter to prevent a zero divide
        :param truncated: True if ReLU is activated for zero targets
        """
        super().__init__()
        self.r = r
        self.truncated = truncated
        self.threshold = threshold

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward path
        :param y: prediction
        :param t: target
        :return: loss
        """
        if self.truncated:
            y = torch.where(t < self.threshold, F.relu(y), y)
        scale = torch.abs(t + self.r)
        y = y / scale
        t = t / scale
        loss = F.mse_loss(y, t)
        return loss