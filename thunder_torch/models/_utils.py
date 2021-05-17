import torch
from typing import List, Union


class View(torch.nn.Module):
    def __init__(self, shape: Union[int, List[int]], channels: int) -> None:
        super().__init__()
        self.shape = shape
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(tuple([x.shape[0]] + [self.channels] + self.shape))


class Reshape(torch.nn.Module):
    def __init__(self, shape: Union[int, List[int]], channels: int) -> None:
        super().__init__()
        self.shape = shape
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, tuple([x.shape[0]] + [self.channels] + self.shape))
