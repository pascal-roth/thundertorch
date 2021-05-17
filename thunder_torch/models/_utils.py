import torch
from typing import Tuple, Union


class View(torch.nn.Module):
    def __init__(self, shape: Union[int, Tuple[int]]) -> None:
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.shape)


class Reshape(torch.nn.Module):
    def __init__(self, shape: Union[int, Tuple[int]]) -> None:
        super().__init__()
        self.shape = shape,

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, self.shape)
