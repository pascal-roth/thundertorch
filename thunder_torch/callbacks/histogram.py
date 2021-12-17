import torch
import os
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only

from thunder_torch import _logger


class Histogram(Callback):

    def __init__(self,  bins: int = 100, filepath: str = 'histograms', boundaries: Optional[List[float]] = None,
                 density: bool = True, title: str = 'Relative Error of Training Data', period: int = 1,
                 monitor: str = 'val_loss', mode: str = 'auto', verbose: int = 0,
                 multi_output: str = 'average') -> None:
        super().__init__()

        # check given filepath
        if os.path.isdir(filepath):
            self.dirpath, self.filename = filepath, 'hist'
        else:
            self.dirpath, self.filename = os.path.split(filepath)
        os.makedirs(self.dirpath, exist_ok=True)

        self.bins = bins
        self.range = boundaries
        self.density = density
        self.title = title
        self.epoch_last_check = None
        self.period = period
        self.monitor = monitor
        self.verbose = verbose
        self.multi_output = multi_output

        torch_inf = torch.tensor(np.Inf)
        mode_dict = {
            'min': (torch_inf, 'min'),
            'max': (-torch_inf, 'max'),
            'auto': (-torch_inf, 'max') if 'acc' in self.monitor or self.monitor.startswith('fmeasure')
            else (torch_inf, 'min'),
        }

        if mode not in mode_dict:
            rank_zero_warn(f'ModelCheckpoint mode {mode} is unknown, '
                           f'fallback to auto mode.', RuntimeWarning)
            mode = 'auto'

        self.best_value, self.mode = mode_dict[mode]

        self.errors_train: Optional[torch.Tensor] = None
        self.errors_val: Optional[torch.Tensor] = None
        self.errors_test: Optional[torch.Tensor] = None

        _logger.info('Histogram creation activated')

    # Utility functions ###############################################################################################
    def check_monitor(self, current: torch.Tensor) -> torch.Tensor:
        if not isinstance(current, torch.Tensor):
            rank_zero_warn(
                f'{current} is supposed to be a torch.Tensor. Saving checkpoint may not work correctly. '
                f'HINT: check the value of {self.monitor} in your validation loop', RuntimeWarning
            )
            current = torch.tensor(current)

        monitor_op = {
            "min": torch.lt,
            "max": torch.gt,
        }[self.mode]

        # If using multiple devices, make sure all processes are unanimous on the decision.
        # should_update_best_and_save = trainer.training_type_plugin.reduce_boolean_decision
        # (should_update_best_and_save)

        _logger.debug(f'Metrics are of device {current.device}')
        if current.device == 'gpu':
            self.best_value.to(current.device)

        return monitor_op(current, self.best_value)

    @rank_zero_only
    def _decide_plot_histogram(self, trainer: pl.Trainer) -> None:
        # only run on main process
        if trainer.proc_rank != 0:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        if self.epoch_last_check is not None and (epoch - self.epoch_last_check) < self.period:
            # skipping in this term
            return

        self.epoch_last_check = epoch

        current = metrics.get(self.monitor)

        if not isinstance(current, torch.Tensor):
            rank_zero_warn(
                f'The metric you returned {current} must be a Torch.Tensor instance, checkpoint not saved '
                f'HINT: what is the value of {self.monitor} in validation_end()?', RuntimeWarning
            )

        if current is None:
            rank_zero_warn(
                f'Can save best model only with {self.monitor} available, skipping.', RuntimeWarning
            )
        elif self.check_monitor(current):
            self.best_value = current
            self._plot_histogram(self.errors_val.numpy(), 'val', 'Validation Data')
            self._plot_histogram(self.errors_train.numpy(), 'train', 'Training Data')
        elif self.verbose > 0:
            _logger.info(f'\nEpoch {epoch:05d}: {self.monitor}  was not best')

    def _plot_histogram(self, rel_errors: np.ndarray, name: str, title: str) -> None:

        # for multiple outputs decide if one plot for each output or the average error should be generated
        if len(rel_errors.shape) > 1 and rel_errors.shape[1] != 1:

            dim2_shape = 1
            for i in range(1, len(rel_errors.shape), 1):
                dim2_shape = dim2_shape * rel_errors.shape[i]
            rel_errors = np.reshape(rel_errors, (len(rel_errors), dim2_shape))

            if self.multi_output == 'average':
                rel_errors = np.mean(np.abs(rel_errors), axis=1)
                # if self.range is not None and self.range[0] < 0:
                #     self.range[0] = 0

            elif self.multi_output == 'single':
                raise NotImplementedError('not yet available')
                # for i in range(rel_errors.shape[1]-1):

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel("Relative Error [%]", size=16)
        ax.set_ylabel('Fraction of Samples', size=16)
        ax.set_title(title, size=20)
        ax.tick_params(axis='both', labelsize=14)

        ax.hist(rel_errors, bins=self.bins, range=self.range, density=self.density)
        ax.axvline(x=0, linestyle="--", linewidth=1, color='grey')

        at = AnchoredText(f'var: {np.var(rel_errors):.2e}\n'
                          f'std: {np.std(rel_errors):.2e}',
                          loc='upper right', frameon=True)
        ax.add_artist(at)

        plt.tight_layout()
        plt.savefig(f'{self.dirpath}/{self.filename}_{name}.jpg')
        plt.close(fig)

    # Model Hooks #####################################################################################################
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            targets = trainer.hiddens["targets"]
            preds = trainer.hiddens["preds"]

            # if len(preds.shape) != 1:
            #     preds = torch.squeeze(preds)

            if self.errors_train is None:
                self.errors_train = ((preds-targets)/(targets+1e-09)) * 100
            else:
                self.errors_train = torch.cat((self.errors_train, ((preds-targets)/(targets+1e-09)) * 100), dim=0)

    # def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    #     self._decide_plot_histogram(trainer, self.errors_train, 'train_histogram', 'Training Data')
    #     self.errors_train = None

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            targets = trainer.hiddens["targets"]
            preds = trainer.hiddens["preds"]

            if self.errors_val is None:
                self.errors_val = ((preds - targets) / (targets + 1e-09)) * 100
            else:
                self.errors_val = torch.cat((self.errors_val, ((preds - targets) / (targets + 1e-09)) * 100), dim=0)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._decide_plot_histogram(trainer)
        self.errors_val = torch.empty(0)
        self.errors_train = torch.empty(0)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]

            if self.errors_test is None:
                self.errors_test = ((preds - targets) / (targets + 1e-09)) * 100
            else:
                self.errors_test = torch.cat((self.errors_test, ((preds - targets) / (targets + 1e-09)) * 100), dim=0)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._plot_histogram(self.errors_test.numpy(), 'test', 'Test Data')
