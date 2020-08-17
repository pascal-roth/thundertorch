import torch as th


class Regularizer(object):

    def reset(self):
        raise NotImplementedError('subclass must implement this method')

    def __call__(self, module, input=None, output=None):
        raise NotImplementedError('subclass must implement this method')


class L1Regularizer(Regularizer):

    def __init__(self, scale=1e-3, module_filter='*'):
        self.scale = float(scale)
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        value = torch.sum(torch.abs(module.weight)) * self.scale
        self.value += value


class L2Regularizer(Regularizer):

    def __init__(self, scale=1e-3, module_filter='*'):
        self.scale = float(scale)
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        value = torch.sum(torch.pow(module.weight,2)) * self.scale
        self.value += value


class L1L2Regularizer(Regularizer):

    def __init__(self, l1_scale=1e-3, l2_scale=1e-3, module_filter='*'):
        self.l1 = L1Regularizer(l1_scale)
        self.l2 = L2Regularizer(l2_scale)
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        self.l1(module, input, output)
        self.l2(module, input, output)
        self.value += (self.l1.value + self.l2.value)


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

class UnitNormRegularizer(Regularizer):
    """
    UnitNorm constraint on Weights

    Constraints the weights to have column-wise unit norm
    """
    def __init__(self,
                 scale=1e-3,
                 module_filter='*'):

        self.scale = scale
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        w = module.weight
        norm_diff = torch.norm(w, 2, 1).sub(1.)
        value = self.scale * torch.sum(norm_diff.gt(0).float().mul(norm_diff))
        self.value += value


class MaxNormRegularizer(Regularizer):
    """
    MaxNorm regularizer on Weights

    Constraints the weights to have column-wise unit norm
    """
    def __init__(self,
                 scale=1e-3,
                 module_filter='*'):

        self.scale = scale
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        w = module.weight
        norm_diff = torch.norm(w,2,self.axis).sub(self.value)
        value = self.scale * torch.sum(norm_diff.gt(0).float().mul(norm_diff))
        self.value += value


class NonNegRegularizer(Regularizer):
    """
    Non-Negativity regularizer on Weights

    Constraints the weights to have column-wise unit norm
    """
    def __init__(self,
                 scale=1e-3,
                 module_filter='*'):

        self.scale = scale
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        w = module.weight
        value = -1 * self.scale * torch.sum(w.gt(0).float().mul(w))
        self.value += value

