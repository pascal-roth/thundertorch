#######################################################################################################################
# Callback to save ckpt under defined filepath
#######################################################################################################################
import torch
import os
import numpy as np
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from pytorch_lightning import _logger as log


class Checkpointing(Callback):

    def __init__(self, filepath: Optional[str] = None, monitor: str = 'val_loss', verbose: bool = False,
                 save_top_k: int = 1, save_weights_only: bool = False,
                 mode: str = 'auto', period: int = 1):
        super().__init__()
        if save_top_k > 0 and filepath is not None and os.path.isdir(filepath) and len(os.listdir(filepath)) > 0:
            rank_zero_warn(
                f"Checkpointing directory {filepath} exists and is not empty with save_top_k != 0."
                "All files in this directory will be deleted when a checkpoint is saved!"
            )
        self._rank = 0

        self.monitor = monitor
        self.verbose = verbose
        if filepath is None:  # will be determined by trainer at runtime
            self.dirpath, self.filename = None, None
        else:
            if os.path.isdir(filepath):
                self.dirpath, self.filename = filepath, '{epoch}'
            else:
                self.dirpath, self.filename = os.path.split(filepath)
            os.makedirs(self.dirpath, exist_ok=True)
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.period = period
        self.epoch_last_check = None
        self.best_k_models = dict({})
        # {filename: monitor}
        self.kth_best_model = ''
        self.best = 0
        self.save_function = None

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

        self.kth_value, self.mode = mode_dict[mode]

    def _del_model(self, filepath: str) -> None:
        if os.path.isfile(filepath):
            os.remove(filepath)

    def _save_model(self, filepath: str) -> None:
        # make paths
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # delegate the saving to the model
        if self.save_function is not None:
            self.save_function(filepath)
        else:
            raise ValueError(".save_function() not set")

    def check_monitor_top_k(self, current: torch.Tensor) -> Union[bool, torch.Tensor]:
        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

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

        return monitor_op(current, self.best_k_models[self.kth_best_model])

    def format_checkpoint_name(self, epoch: int, ver: int = None) -> str:
        """
        Generate a filename according to the defined template.
        """
        if self.filename is None:
            filename = f'ckpt_epoch_{epoch}'
        elif self.save_top_k != 1:
            filename = f'{self.filename}_epoch_{epoch}'
        else:
            filename = self.filename

        str_ver = f'_v{ver}' if ver is not None else ''
        filepath = os.path.join(self.dirpath, filename + str_ver + '.ckpt')
        return filepath

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # only run on main process
        if trainer.proc_rank != 0:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epoch_last_check is not None and (epoch - self.epoch_last_check) < self.period:
            # skipping in this term
            return

        self.epoch_last_check = epoch

        filepath = self.format_checkpoint_name(epoch)
        version_cnt = 0
        while os.path.isfile(filepath):
            filepath = self.format_checkpoint_name(epoch, ver=version_cnt)
            # this epoch called before
            version_cnt += 1

        if self.save_top_k != -1:
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
            elif self.check_monitor_top_k(current):
                self._do_check_save(filepath, current, epoch)
            elif self.verbose > 0:
                log.info(f'\nEpoch {epoch:05d}: {self.monitor}  was not in top {self.save_top_k}')

        else:
            if self.verbose > 0:
                log.info(f'\nEpoch {epoch:05d}: saving model to {filepath}')
            self._save_model(filepath)

    def _do_check_save(self, filepath: str, current: torch.tensor, epoch: int) -> None:
        # remove kth

        del_list = []
        if len(self.best_k_models) == self.save_top_k and self.save_top_k > 0:
            delpath = self.kth_best_model
            self.best_k_models.pop(self.kth_best_model)
            del_list.append(delpath)

        self.best_k_models[filepath] = current
        if len(self.best_k_models) == self.save_top_k:
            # monitor dict has reached k elements
            _op = max if self.mode == 'min' else min
            self.kth_best_model = _op(self.best_k_models,
                                      key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model]

        _op = min if self.mode == 'min' else max
        self.best = _op(self.best_k_models.values())

        if self.verbose > 0:
            log.info(
                f'\nEpoch {epoch:05d}: {self.monitor} reached'
                f' {current:0.5f} (best {self.best:0.5f}), saving model to'
                f' {filepath} as top {self.save_top_k}')
        self._save_model(filepath)

        for cur_path in del_list:
            if cur_path != filepath:
                self._del_model(cur_path)


