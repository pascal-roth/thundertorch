import torch
from collections.abc import Mapping, Sequence

from typing import Any, Callable, Optional, Union


METRIC_EPS = 1e-6


def dim_zero_cat(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(x, dim=0)


def dim_zero_sum(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=0)


def dim_zero_mean(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=0)


def _flatten(x: list) -> list:
    return [item for sublist in x for item in sublist]


def to_onehot(
        tensor: torch.Tensor,
        num_classes: int,
) -> torch.Tensor:
    """
    Converts a dense label tensor to one-hot format
    Args:
        tensor: dense label tensor, with shape [N, d1, d2, ...]
        num_classes: number of classes C
    Output:
        A sparse label tensor with shape [N, C, d1, d2, ...]
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> to_onehot(x, num_classes=4)
        tensor([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    """
    dtype, device, shape = tensor.dtype, tensor.device, tensor.shape
    tensor_onehot = torch.zeros(shape[0], num_classes, *shape[1:],
                                dtype=dtype, device=device)
    index = tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def _check_same_shape(pred: torch.Tensor, target: torch.Tensor) -> None:
    """ Check that predictions and target have the same shape, else raise error """
    if pred.shape != target.shape:
        raise RuntimeError('Predictions and targets are expected to have the same shape')


def apply_to_collection(data: Any, dtype: Union[type, tuple], function: Callable, *args: Optional[Any],
                        **kwargs: Optional[Any]) -> Any:
    """
    Recursively applies a function to all elements of a certain dtype.
    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)
    Returns:
        the resulting collection
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    elif isinstance(data, Mapping):
        return elem_type({k: apply_to_collection(v, dtype, function, *args, **kwargs)
                          for k, v in data.items()})
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # named tuple
        return elem_type(*(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([apply_to_collection(d, dtype, function, *args, **kwargs) for d in data])

    # data is neither of dtype, nor a collection
    return data


def gather_all_tensors_if_available(result: Union[torch.Tensor], group: Optional[Any] = None) -> torch.tensor:
    """
    Function to gather all tensors from several ddp processes onto a list that
    is broadcasted to all processes
    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)
    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD

        world_size = torch.distributed.get_world_size(group)

        gathered_result = [torch.zeros_like(result) for _ in range(world_size)]

        # sync and broadcast all
        torch.distributed.barrier(group=group)
        torch.distributed.all_gather(gathered_result, result, group)

        result = gathered_result
    return result
