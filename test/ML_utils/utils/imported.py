import torch
from argparse import Namespace
from thunder_torch.models.ModelBase import LightningModelBase

class LightningFlexMLPImported(LightningModelBase):
    """
    Example for a self written model that is imported

    Hyperparameters of the model
    ----------------------------
    - inputs:                int         Input dimension (required)
    - outputs:                int         Output dimension (required)
    - number_hidden_layers:         list        List of hidden layers with number of hidden neurons as layer entry (required)
    - activation:           str         activation fkt that is included in torch.nn (default: ReLU)
    - loss:                 str         loss fkt that is included in torch.nn (default: MSELoss)
    - optimizer:            dict        dict including optimizer fkt type and possible parameters, optimizer has to be
                                        included in torch.optim (default: {'type': Adam, 'params': {'lr': 1e-3}})
    - scheduler:            dict        dict including execute flag, scheduler fkt type and possible parameters, scheduler
                                        has to be included in torch.optim.lr_scheduler (default: {'execute': False})
    - num_workers:          int         number of workers in DataLoaders (default: 10)
    - batch:                int         batch size of DataLoaders (default: 64)
    - output_activation:    str         activation fkt  (default: False)
    """

    def __init__(self, hparams: Namespace) -> None:
        """
        Initializes a flexMLP model based on the provided parameters

        Parameters
        ----------
        hparams         - Namespace object including hyperparameters
        """
        super().__init__()

        self.hparams = hparams
        self.get_default()
        self.get_functions()
        self.min_val_loss = None

        # Construct MLP with a variable number of hidden layers
        self.layers = []
        self.construct_mlp(self.hparams.inputs, self.hparams.number_hidden_layers, self.hparams.outputs)

        if hasattr(self.hparams, 'output_activation'):
            self.layers.append(getattr(torch.nn, self.hparams.output_activation)())

        self.layers = torch.nn.Sequential(*self.layers)
