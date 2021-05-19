import torch
from typing import List, Union, Tuple


class View(torch.nn.Module):
    """
    Returns a new tensor with the same data as the self tensor but of a different shape
    """
    def __init__(self, shape: Union[int, List[int]], channels: int) -> None:
        super().__init__()
        self.shape = shape if not isinstance(shape, int) else [shape]
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.view(tuple([x.shape[0]] + [self.channels] + self.shape))


class Reshape(torch.nn.Module):
    """
    Returns a tensor with the same data and number of elements as input, but with the specified shape. When possible,
    the returned tensor will be a view of input. Otherwise, it will be a copy. Contiguous inputs and inputs with
    compatible sides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.

    A single dimension may be -1, in which case it’s inferred from the remaining dimensions and the number of
    elements in input.
    """
    def __init__(self, shape: Union[int, List[int]], channels: int) -> None:
        super().__init__()
        self.shape = shape
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.reshape(x, tuple([x.shape[0]] + [self.channels] + self.shape))


class Cat(torch.nn.Module):
    """
    Concatenates the given sequence of tensors in the given dimension. All tensors must either have the same shape
    (except in the concatenating dimension) or be empty.
    """
    def __init__(self, dim: int = 0) -> None:
        """
        Parameters
        ----------
        dim                         - Dimension where the tensors should be concatenated
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        return torch.cat(x, self.dim)


class Split(torch.nn.Module):
    """
    Splits the tensor into chunks. Each chunk is a view of the original tensor.

    If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible).
    Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.

    If split_size_or_sections is a list, then tensor will be split into len(split_size_or_sections) chunks with sizes
    in dim according to split_size_or_sections
    """
    def __init__(self, split_size_or_sections: Union[int, List[int]], dim: int = 0) -> None:
        """
        Parameters
        ----------
        split_size_or_sections      - size of a single chunk or list of sizes for each chunk
        dim                         - dimension along which to split the tensor
        """
        super().__init__()
        self.dim = dim
        self.split_size_or_sections = split_size_or_sections

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:  # type: ignore[override]
        """
        Parameters
        ----------
        x                           - tensor to split
        """
        return torch.split(x, self.split_size_or_sections, self.dim)
