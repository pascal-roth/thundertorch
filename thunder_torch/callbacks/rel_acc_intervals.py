import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from typing import Optional, Any
from pytorch_lightning.callbacks import Callback

from thunder_torch import metrics
from thunder_torch import _logger


class RelIntervals(Callback):

    def __init__(self, rel_threshold: list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5], path: str = 'intervals',
                 **kwargs: Any) -> None:
        super().__init__()

        self.rel_thresholds = rel_threshold
        self.path = path

        self.intervals_rel_acc_train = []
        self.intervals_rel_acc_val = []
        self.intervals_rel_acc_test = []

        for i in range(len(rel_threshold)):
            self.intervals_rel_acc_train.append(metrics.RelIntervals(rel_threshold[i], **kwargs))
            self.intervals_rel_acc_val.append(metrics.RelIntervals(rel_threshold[i], **kwargs))
            self.intervals_rel_acc_test.append(metrics.RelIntervals(rel_threshold[i], **kwargs))

        _logger.info(f'RelInterval Metrics initialized with rel_threshold: {rel_threshold}')

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.max_epochs - 1 == trainer.current_epoch) and hasattr(trainer, 'hiddens'):
            targets = trainer.hiddens["targets"]
            preds = trainer.hiddens["preds"]

            for i in range(len(self.intervals_rel_acc_train)):
                self.intervals_rel_acc_train[i](preds, targets)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        accuracies = np.zeros((len(self.intervals_rel_acc_train), 5))

        for i in range(len(self.intervals_rel_acc_train)):
            accuracies[i, 0] = -self.rel_thresholds[i]
            accuracies[i, 1] = self.rel_thresholds[i]
            accuracies[i, 2:] = self.intervals_rel_acc_train[i].compute()

        print(f"****** RELATIVE ACCURACY INTERVALS ****** \n"
              f"{accuracies} \n")

        self.plot_intervals(accuracies, 'train_intervals')

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.max_epochs - 1 == trainer.current_epoch) and hasattr(trainer, 'hiddens'):
            targets = trainer.hiddens["targets"]
            preds = trainer.hiddens["preds"]

            for i in range(len(self.intervals_rel_acc_train)):
                self.intervals_rel_acc_val[i](preds, targets)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.max_epochs - 1 == trainer.current_epoch:
            accuracies = np.zeros((len(self.intervals_rel_acc_val), 5))

            for i in range(len(self.intervals_rel_acc_val)):
                accuracies[i, 0] = -self.rel_thresholds[i]
                accuracies[i, 1] = self.rel_thresholds[i]
                accuracies[i, 2:] = self.intervals_rel_acc_val[i].compute()

            print(f"****** RELATIVE ACCURACY INTERVALS ****** \n"
                  f"{accuracies} \n")

            self.plot_intervals(accuracies, 'val_intervals')

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(trainer, 'hiddens'):
            preds = trainer.hiddens["preds"]
            targets = trainer.hiddens["targets"]

            for i in range(len(self.intervals_rel_acc_train)):
                self.intervals_rel_acc_test[i](preds, targets)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        accuracies = np.zeros((len(self.intervals_rel_acc_test), 5))

        for i in range(len(self.intervals_rel_acc_test)):
            accuracies[i, 0] = -self.rel_thresholds[i]
            accuracies[i, 1] = self.rel_thresholds[i]
            accuracies[i, 2:] = self.intervals_rel_acc_test[i].compute()

        print(f"****** RELATIVE ACCURACY INTERVALS ****** \n"
              f"{accuracies} \n")

        self.plot_intervals(accuracies, 'test_intervals')

    def plot_intervals(self, accuracies: np.ndarray, name: str) -> None:
        fig, ax = plt.subplots()

        ind = np.arange(len(accuracies))

        ax.bar(ind, accuracies[:, 2] * 100, width=0.5, label="lower boundary")
        ax.bar(ind, accuracies[:, 3] * 100, width=0.5, label="upper boundary", bottom=accuracies[:, 2] * 100)

        ax.set_ylabel('% of samples within tolerance range')
        ax.set_xlabel(r'tolerance range ($\pm$) in %')
        ax.set_xticks(ind)
        ax.set_xticklabels(accuracies[:, 1] * 100)
        ax.legend()

        plt.savefig(f'{os.getcwd()}/{self.path}/{name}.jpg')
        np.savetxt(f'{os.getcwd()}/{self.path}/{name}.csv', accuracies)
