from pytorch_lightning import Callback


class SaveModelWithCustomFunction(Callback):
    """
    This callback saves the best validation model
    """
    def __init__(self, save_function):
        """

        Parameters
        ----------
        save_function - custom function that saves the model
        """
        self.best_loss = 0

    def on_validation_end(self, trainer, pl_module):
        print('do something when training ends')
