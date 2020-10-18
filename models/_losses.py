import torch
import torch.nn.functional as F


class RelativeMSELoss:
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
