import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Optional, Any

from thunder_torch import metrics
from thunder_torch import _logger


class AbsRelAccuracy(Callback):

    def __init__(self, abs_threshold: float = 0.005, rel_threshold: float = 0.01, **kwargs: Optional[Any]):
        super().__init__()

        self.abs_rel_acc_train = metrics.AbsRelAccuracy(abs_threshold, rel_threshold, **kwargs)
        self.abs_rel_acc_val = metrics.AbsRelAccuracy(abs_threshold, rel_threshold, **kwargs)
        self.abs_rel_acc_test = metrics.AbsRelAccuracy(abs_threshold, rel_threshold, **kwargs)

        _logger.debug(f'AbsRelAccuracy Metrics Initialized with abs_threshold: {abs_threshold}, rel_threshold: '
                      f'{rel_threshold}')

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            targets = trainer.hiddens["targets"]
            preds = trainer.hiddens["preds"]
            self.abs_rel_acc_train(preds, targets)

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        abs_acc, rel_acc, abs_rel_acc = self.abs_rel_acc_train.compute()
        acc_dict = {'train_abs_acc': abs_acc, 'train_rel_acc': rel_acc, 'train_abs_rel_acc': abs_rel_acc}
        if trainer.logger:
            trainer.logger.log_metrics(acc_dict, step=trainer.global_step)
        trainer.add_progress_bar_metrics(acc_dict)
        # trainer.logger_connector.add_progress_bar_metrics(acc_dict)

        # pl_module.log('train_abs_acc', abs_acc, logger=True, prog_bar=True)
        # pl_module.log('train_rel_acc', rel_acc, logger=True, prog_bar=True)
        # pl_module.log('train_abs_rel_acc', abs_rel_acc, logger=True, prog_bar=True)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]
            self.abs_rel_acc_val(preds, targets)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        abs_acc, rel_acc, abs_rel_acc = self.abs_rel_acc_val.compute()
        acc_dict = {'val_abs_acc': abs_acc, 'val_rel_acc': rel_acc, 'val_abs_rel_acc': abs_rel_acc}

        # if not trainer.running_sanity_check:
        if trainer.logger:
            trainer.logger.log_metrics(acc_dict, step=trainer.global_step)
        trainer.add_progress_bar_metrics(acc_dict)
        # trainer.logger_connector.add_progress_bar_metrics(acc_dict)

        # pl_module.log('val_abs_acc', abs_acc, logger=True, prog_bar=True)
        # pl_module.log('val_rel_acc', rel_acc, logger=True, prog_bar=True)
        # pl_module.log('val_abs_rel_acc', abs_rel_acc, logger=True, prog_bar=True)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]
            self.abs_rel_acc_test(preds, targets)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        abs_acc, rel_acc, abs_rel_acc = self.abs_rel_acc_test.compute()
        acc_dict = {'test_abs_acc': abs_acc.item(), 'test_rel_acc': rel_acc.item(),
                    'test_abs_rel_acc': abs_rel_acc.item()}
        if trainer.logger:
            trainer.logger.log_metrics(acc_dict, step=trainer.global_step)
        print(acc_dict)
        trainer.add_progress_bar_metrics(acc_dict)
        # trainer.logger_connector.add_progress_bar_metrics(acc_dict)

        # pl_module.log('test_abs_acc', abs_acc, logger=True, prog_bar=True)
        # pl_module.log('test_rel_acc', rel_acc, logger=True, prog_bar=True)
        # pl_module.log('test_abs_rel_acc', abs_rel_acc, logger=True, prog_bar=True)
