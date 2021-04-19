"""
 Thunder Torch - a collection of useful routines for machine learning applications

 author: Pascal Roth
 email:  roth.pascal@outlook.de

"""

# Initialize logger for Thunder Torch
import logging as python_logging
import sys

_logger = python_logging.getLogger("ml_utils")
_logger.setLevel(python_logging.INFO)

console_handler = python_logging.StreamHandler(sys.stdout)
console_handler.setFormatter(python_logging.Formatter("%(module)s: %(message)s"))
_logger.addHandler(console_handler)

# Module lists
_modules_activation = ['torch.nn']
_modules_loss = ['torch.nn', 'thunder_torch.models']
_modules_optim = ['torch.optim']
_modules_lr_scheduler = ['torch.optim.lr_scheduler']
_modules_models = ['thunder_torch.models']
_modules_loader = ['thunder_torch.loader']
_modules_callbacks = ['pytorch_lightning.callbacks', 'thunder_torch.callbacks']

# Check imports
try:
    import torch
    torch.set_default_dtype(torch.float64)

    # Make sure right matplotlib backend is running
    import matplotlib
    import platform
    if platform.system() == "Darwin":
        matplotlib.use("MacOSX")

except ImportError as error:
    # Check if thunder_torch was installed with ML support
    print(error.__class__.__name__ + ": " + error.msg)
    print("Are you sure you have installed the thunder_torch with ML support?")
    print("Run InstallThunderTorch.sh again and answer with 'y' when asked for ML support!")
