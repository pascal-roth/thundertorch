import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from thunder_torch import metrics


class Explained_Variance(Callback):

    def on_init_end(self, trainer: pl.Trainer) -> None:
        self.explained_variance_train = metrics.ExplainedVariance()
        self.explained_variance_val = metrics.ExplainedVariance()
        self.explained_variance_test = metrics.ExplainedVariance()

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            # inputs = trainer.hiddens["inputs"]
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]
            self.explained_variance_train(preds, targets)

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        train_ExpVar = self.explained_variance_train.compute()
        pbar = {'train_ExpVar': train_ExpVar}
        trainer.add_progress_bar_metrics(pbar)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]
            self.explained_variance_val(preds, targets)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pbar = {'val_ExpVar': self.explained_variance_val.compute()}
        trainer.add_progress_bar_metrics(pbar)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]
            self.explained_variance_test(preds, targets)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        test_ExpVar = self.explained_variance_test.compute()
        pbar = {'test_ExpVar': test_ExpVar}
        trainer.add_progress_bar_metrics(pbar)
