from typing import Union, Tuple, Sequence

import torch

from thunder_torch.metrics.utils import _check_same_shape


def _explained_variance_update(preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_same_shape(preds, target)
    return preds, target


def _explained_variance_compute(preds: torch.Tensor,
                                target: torch.Tensor,
                                multioutput: str = 'uniform_average',
                                ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    diff_avg = torch.mean(target - preds, dim=0)
    numerator = torch.mean((target - preds - diff_avg) ** 2, dim=0)

    target_avg = torch.mean(target, dim=0)
    denominator = torch.mean((target - target_avg) ** 2, dim=0)

    # Take care of division by zero
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = torch.ones_like(diff_avg)
    output_scores[valid_score] = 1.0 - float(numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

    # Decide what to do in multioutput case
    # Todo: allow user to pass in tensor with weights
    if multioutput == 'raw_values':
        return output_scores
    elif multioutput == 'uniform_average':
        return torch.mean(output_scores)
    elif multioutput == 'variance_weighted':
        denom_sum = torch.sum(denominator)
        return torch.sum(denominator / denom_sum * output_scores)
    else:
        raise ValueError('value for multioutput not defined')


def explained_variance(preds: torch.Tensor,
                       target: torch.Tensor,
                       multioutput: str = 'uniform_average',
                       ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """
    Computes explained variance.
    Args:
        pred: estimated labels
        target: ground truth labels
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is `'uniform_average'`.):
            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances
    Example:
        >>> from thunder_torch.metrics.functional import explained_variance
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> explained_variance(preds, target)
        tensor(0.9572)
        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> explained_variance(preds, target, multioutput='raw_values')
        tensor([0.9677, 1.0000])
    """
    preds, target = _explained_variance_update(preds, target)
    return _explained_variance_compute(preds, target, multioutput)
