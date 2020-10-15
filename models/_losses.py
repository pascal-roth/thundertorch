import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeMSELoss(nn.Module):
    # """
    # Relative loss function based on MSE loss
    # """
    # def __init__(self, **kwargs) -> None:
    #     """
    #     :param r: residual parameter to prevent a zero divide
    #     :param truncated: True if ReLU is activated for zero targets
    #     """
    #     super().__init__()
    #     self.r = kwargs.pop('r', 0.01)
    #     self.truncated = kwargs.pop('truncated', False)
    #     self.threshold = kwargs.pop('threshold', 1e-6)
    #
    # def forward(self, y, t) -> torch.Tensor:
    #     """
    #     Forward path
    #     :param y: prediction
    #     :param t: target
    #     :return: loss
    #     """
    #     if self.truncated:
    #         y = torch.where(t < self.threshold, F.relu(y), y)
    #     scale = torch.abs(t + self.r)
    #     y = y / scale
    #     t = t / scale
    #     loss = F.mse_loss(y, t)
    #     return loss

    @staticmethod
    def loss_fn(y_hat, y):
        r = 0.01
        truncated = False
        threshold = 1e-6

        if truncated:
            y_hat = torch.where(y < threshold, F.relu(y_hat), y_hat)
        scale = torch.abs(y + r)
        y_hat = y_hat / scale
        y = y / scale
        loss = F.mse_loss(y_hat, y)

        return loss
