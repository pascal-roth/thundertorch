import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from thunder_torch import metrics
from thunder_torch import _logger


class RelError(Callback):

    def __init__(self, **kwargs):
        super().__init__()

        self.rel_error_train = metrics.RelError(**kwargs)
        self.rel_error_val = metrics.RelError(**kwargs)
        self.rel_error_test = metrics.RelError(**kwargs)

        _logger.info('RelError metric activated')

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if hasattr(trainer, 'hiddens'):
            targets = trainer.hiddens["targets"]
            preds = trainer.hiddens["preds"]
            self.rel_error_train(preds, targets)

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if hasattr(trainer, 'hiddens'):
            mean_rel_error, mean_abs_error = self.rel_error_train.compute()
            acc_dict = {'rel_error': mean_rel_error, 'abs_error': mean_abs_error}
            if trainer.logger:
                trainer.logger.log_metrics(acc_dict, step=trainer.global_step)
            trainer.add_progress_bar_metrics(acc_dict)
            # trainer.logger_connector.add_progress_bar_metrics(acc_dict)

            # pl_module.log('train_abs_acc', abs_acc, logger=True, prog_bar=True)
            # pl_module.log('train_rel_acc', rel_acc, logger=True, prog_bar=True)
            # pl_module.log('train_abs_rel_acc', abs_rel_acc, logger=True, prog_bar=True)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if hasattr(trainer, 'hiddens'):
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]
            self.rel_error_val(preds, targets)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if hasattr(trainer, 'hiddens'):
            mean_rel_error, mean_abs_error = self.rel_error_val.compute()
            acc_dict = {'val_rel_error': mean_rel_error, 'val_abs_error': mean_abs_error}

            # if not trainer.running_sanity_check:
            if trainer.logger:
                trainer.logger.log_metrics(acc_dict, step=trainer.global_step)
            trainer.add_progress_bar_metrics(acc_dict)
            # trainer.logger_connector.add_progress_bar_metrics(acc_dict)

            # pl_module.log('val_abs_acc', abs_acc, logger=True, prog_bar=True)
            # pl_module.log('val_rel_acc', rel_acc, logger=True, prog_bar=True)
            # pl_module.log('val_abs_rel_acc', abs_rel_acc, logger=True, prog_bar=True)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if hasattr(trainer, 'hiddens'):
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]
            self.rel_error_test(preds, targets)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if hasattr(trainer, 'hiddens'):
            mean_rel_error, mean_abs_error = self.rel_error_test.compute()
            acc_dict = {'test_rel_error': mean_rel_error.item(), 'test_abs_error': mean_abs_error.item()}
            if trainer.logger:
                trainer.logger.log_metrics(acc_dict, step=trainer.global_step)
            print(acc_dict)
            trainer.add_progress_bar_metrics(acc_dict)
            # trainer.logger_connector.add_progress_bar_metrics(acc_dict)

            # pl_module.log('test_abs_acc', abs_acc, logger=True, prog_bar=True)
            # pl_module.log('test_rel_acc', rel_acc, logger=True, prog_bar=True)
            # pl_module.log('test_abs_rel_acc', abs_rel_acc, logger=True, prog_bar=True)
