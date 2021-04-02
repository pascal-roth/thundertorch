"""
 ML_Utils - a collection of useful routines for machine learning applications

 author: Pascal Roth, Julian Bissantz
 email:  roth.pascal@outlook.de, bissantz@stfs.tu-darmstadt.de

"""

# Initialize logger for ML_utlits
import logging as python_logging
import sys

_logger = python_logging.getLogger("ml_utils")
_logger.setLevel(python_logging.INFO)

console_handler = python_logging.StreamHandler(sys.stdout)
console_handler.setFormatter(python_logging.Formatter("%(module)s: %(message)s"))
_logger.addHandler(console_handler)

# Module lists
_modules_activation = ['torch.nn']
_modules_loss = ['torch.nn', 'stfs_pytoolbox.ML_Utils.models']
_modules_optim = ['torch.optim']
_modules_lr_scheduler = ['torch.optim.lr_scheduler']
_modules_models = ['stfs_pytoolbox.ML_Utils.models']
_modules_loader = ['stfs_pytoolbox.ML_Utils.loader']
_modules_callbacks = ['pytorch_lightning.callbacks', 'stfs_pytoolbox.ML_Utils.callbacks']

# Check imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_default_dtype(torch.float64)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    from .models import *
    from .loader import *
    from .models._losses import *
    from .models._regularizers import *

    import os
    import sys
    from sklearn.preprocessing import MinMaxScaler

    # Make sure right matplotlib backend is running
    import matplotlib
    import platform
    if platform.system() == "Darwin":
        matplotlib.use("MacOSX")

except ImportError as error:
    # Check if stfs_pytoolbox was installed with ML support
    print(error.__class__.__name__ + ": " + error.msg)
    print("Are you sure you have installed the stfs_pytoolbox with ML support?")
    print("Run Install_stfs_pytoolbox.sh again and answer with 'y' when asked for ML support!")