import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class WeightInitializer(Callback):
    """
    Initializes the weights of the last layer (containing weights, in case last layer is an activation function)
    to a given value
    """
    def __init__(self, value: float = 0.1) -> None:
        super().__init__()
        self.value = value

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for layer in reversed(pl_module.layers):
            if hasattr(layer, "weight"):
                layer.weight.data. fill_(self.value)
                break
