Metrics
=======

Evaluating the machine learning algorithm is an essential part of any
project. The implemented model may give satisfying results when
evaluated using a metric say accuracy_score but may give poor results
when evaluated against other metrics such as logarithmic_loss or any
other such metric. Often one metric such as accuracy for classification
tasks is used to measure the performance of our model, however it is not
enough to truly judge our model. As a consequence, the latest versions
of PyTorch Lightning have a metrics API, which can basically also be
used with the toolbox version of lightning (limited due to cluster
support for PyTorch only till version 1.2.0). A detailed documentation
of the Metrics API can be found
`here <https://pytorch-lightning.readthedocs.io/en/latest/metrics.html>`__.

A minimal example of an accuracy metrics is as follows:

.. code:: python

    from stfs_pytoolbox.ML_Utils.metrics.metric import Metric
    
    class MyAccuracy(Metric):
        def __init__(self, dist_sync_on_step=False):
            super().__init__(dist_sync_on_step=dist_sync_on_step)
    
            self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
        def update(self, preds: torch.Tensor, target: torch.Tensor):
            preds, target = self._input_format(preds, target)
            assert preds.shape == target.shape
    
            self.correct += torch.sum(preds == target)
            self.total += target.numel()
    
        def compute(self):
            return self.correct.float() / self.total

**Usage in the Toolbox**

The Toolbox already included the metric base class of Lightning and the
corresponding utils functions. In order to use the pre-implemented
metrics, the source code has to be copied into the metrics directory of
the ML_Utils toolbox. As an example, this has been made with the
“Explained Variance” metric.

Metrics usage
-------------

The metrics can be employed by implementing the different steps directly
in the model:

.. code:: python

    def __init__(self):
        ...
        self.accuracy = pl.metrics.Accuracy()
    
    def training_step(self, batch, batch_idx):
        logits = self(x)
        ...
        # log step metric
        self.log('train_acc_step', self.accuracy(logits, y))
        ...
    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())

However, it is recommended to implement the metric as a **Callback**.
Therefore, the model stays clean and different metrics can be activated
just by changing a key in the yaml template. An example how to implement
a metric as callback can be found `here <./Callbacks.html>`__.
