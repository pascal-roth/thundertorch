"""
 ML_Utils - a collection of useful routines for machine learning applications
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    from .models import *
    from .losses import *
    from .regularizers import *

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


def trainFlexMLP(model, path, features, labels, df_train, df_validation=None, epochs=10,
                      batch=16, lr=0.001, loss_fn = torch.nn.MSELoss(), plot=False, scalers=[],  use_scheduler=False):
    """Optimize the weights of a given MLP.

    Parameters
    ----------
    model - SimpleMLP : model to optimize
    path - String : path to save best model weights
    features - list of strings : list of features
    labels - list of strings : list of labels
    df_train - pd.DataFrame : DataFrame which contains all training features and labels
    df_validation - pd.DataFrame : DataFrame which contains all validation features and labels
    epochs - Integer : number of epochs to train
    batch - Integer : size of training batch
    l_rate - Float : learning rate
    plot: Bool
        plots loss curves

    Note: The training and validation data will be scaled in this function, therefore no prior scaling is needed

    Returns
    -------
    model - SimpleMLP : opimized model in evaluation mode
    training_loss - List : training loss developments over epochs
    validation_loss - List : validation loss developments over epochs
    """

    def updateLines(ax, trainloss, valloss):
        """
        updates line values for loss plot

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axis object to be updated
        trainloss: list of floats
        valloss: list of floats
        """
        x = list(range(1, len(trainloss)+1))
        lines = ax.get_lines()
        lines[0].set_xdata(x)
        lines[1].set_xdata(x)
        lines[0].set_ydata(trainloss)
        lines[1].set_ydata(valloss)
        #ax.relim()
        plt.draw()
        plt.pause(1e-17)

    # if validation data is None
    if df_validation is None:
        df_validation = df_train

    # extract training and validation features and labels from dataframes
    x_train = df_train[features].copy()
    y_train = df_train[labels].copy()

    x_validation = df_validation[features].copy()
    y_validation = df_validation[labels].copy()

    # scale training and validation data

    # check if min and max value are equal, if yes fit scaler with min value of that quantity set to 0
    # MinMaxScaler can actually handle this but returning a value of 0 if min ==max
    # but in order to scale appropriately manually in AssemblyModel min value is set to 0
    x_train_min = np.amin(x_train.values, axis=0)
    x_train_max = np.amax(x_train.values, axis=0)
    diff = x_train_max - x_train_min
    idx = np.where(np.isin(diff, 0))

    for i in idx:
        x_train_min[i] = 0

    if not scalers:
        # scale features with new min and max values and labels with all values
        featureScaler = MinMaxScaler()
        featureScaler.fit(np.stack((x_train_max, x_train_min), axis=0))
        labelScaler = MinMaxScaler()
        labelScaler.fit(y_train.values)
    else:
        featureScaler, labelScaler = scalers


    x_train = featureScaler.transform(x_train.values)
    y_train = labelScaler.transform(y_train.values)
    x_validation = featureScaler.transform(x_validation.values)
    y_validation = labelScaler.transform(y_validation.values)

    # convert numpy arrays to tensor
    x_train_tensor = torch.from_numpy(x_train.astype(np.float64))
    y_train_tensor = torch.from_numpy(y_train.astype(np.float64))

    x_validation_tensor = torch.from_numpy(x_validation.astype(np.float64))
    y_validation_tensor = torch.from_numpy(y_validation.astype(np.float64))

    # Create training dataloader
    trainset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)

    # track losses, calculate initial validation loss and set that as baseline
    prediction = model.forward(x_validation_tensor)
    best_loss = loss_fn(prediction, y_validation_tensor).item()
    print("\nInitial validation loss is: {:6.5e}\n".format(best_loss))
    train_losses, validation_losses = [best_loss], [best_loss]

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-8)
        scheduler.step(best_loss)

    # if training on gpu
    if torch.cuda.is_available():
        device="cuda:0"
    else:
        device="cpu"

    print("Training on {}!".format(device))

    model.to(device)

    # Prepare plot of loss curves
    if plot:
        # print("backend: "+plt.get_backend())
        xdata = [0]
        plt.show()
        ax = plt.gca()
        ax.set_xlim(0, epochs)
        ax.set_ylim(best_loss/1e7, best_loss*2)
        plt.yscale("log")
        ax.plot(xdata, train_losses, 'r-', label="Training loss")
        ax.plot(xdata, validation_losses, 'b-', label="Validation loss")
        ax.legend()
        plt.draw()
        plt.pause(1e-17)

    # prepare tqdm progress bars
    # nice example usage can be found here:
    # https://medium.com/@philipplies/progress-bar-and-status-logging-in-python-with-tqdm-35ce29b908f5

    outer = tqdm(total=epochs, position=0)
    inner = tqdm(total=int(len(trainloader.dataset) / trainloader.batch_size), position=1)
    best_log = tqdm(total=0, position=4, bar_format='{desc}')

    # for epoch in tqdm(range(1,epochs+1)):
    for epoch in range(1,epochs+1):
        running_loss = 0
        model.train()

        # get training batch
        for batch in iter(trainloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()             # Empty gradients
            prediction = model.forward(x)     # model prediction on batch
            loss = loss_fn(prediction, y)     # calculate loss of batch
            loss.backward()                   # calculate gradient of loss
            optimizer.step()                  # update model parameters

            running_loss += loss.item()
            inner.update(1)
        # Reuse inner progress bar
        outer.update(1)
        inner.refresh()  # flush last output
        inner.reset()   # reset progress bar

        # Track training loss
        train_losses.append(running_loss/len(trainloader))

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            pred = model(x_validation_tensor)
            val_loss = loss_fn(pred, y_validation_tensor).item()
            validation_losses.append(val_loss)
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                createFlexMLPCheckpoint(model, path, features=features, labels=labels, epochs=epoch, scalers=[featureScaler, labelScaler])
        if plot:
            updateLines(ax, train_losses, validation_losses)
        if use_scheduler:
            scheduler.step(val_loss)

        # write output using progress bars
        outer.write("Epoch: {:05d}/{:05d}, Training loss: {:6.5e}, Validation loss: {:6.5e}".format(epoch, epochs, train_losses[-1], validation_losses[-1]))
        best_log.set_description_str("Best validation loss: {:6.5e}\tCurrent lr: {:6.5e}".format(best_loss, optimizer.param_groups[0]['lr']))

    # close progress bars
    inner.close()
    outer.close()

    return model.eval(), train_losses, validation_losses


def scale_df(df, scaler=None):
    """
    Scales the data in pandas.DataFrame
    Parameters
    ----------
    df: pandas.DataFrame
        data to be scaled
    scaler: sklearn.scaler
        If no scaler is provided a MinxMaxScaler is created

    Returns
    -------
    numpy.ndarry of scaled data from DataFrame
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df.values)

    scaled_data = scaler.transform(df.values)

    return scaled_data


def unscale_df(df, labels, scaler):
    """
    Unscales the data in pandas.DataFrame
    Parameters
    ----------
    df: pandas.DataFrame
        data to be unscaled
    labels: list str
        list of strings for column names
    scaler: sklearn.scaler
        scaler to unscale values

    Returns
    -------
    pd.DataFrame of unscaled data
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df.values)

    scaled_data = scaler.transform(df.values)

    return scaled_data


def runFlexMLP(model, data, features=None, labels=None, scalers=None):
    """
    Runs a FlexMLP model with some data.
    This data is scaled, the model is run and the rescaled prediciton is returned as pandas.DataFrame

    Parameters
    ----------
    model: FlexMLP or path to load a model
    data: pandas.df
        data that contains model input

    optional Arguments
    ------------------
    features: list str
        list of features/model inputs
    labels: list str
        list of labels/model output
    scalers: sklearn.Scales
        list of features and labelscaler to scale the data

    Returns
    -------
    pandas.Dataframe with result data
    """

    if isinstance(model, str):
        model, features, labels, _, [featureScaler, labelScaler] = loadFlexMLPCheckpoint(model)

    elif isinstance(model, FlexMLP):
        # Do nothing
        if features is None or labels is None:
            print("Error: list of features and labels must be provided if model is FlexMLP object")
        if scalers is not None:
            [featureScaler, labelScaler] = scalers
    else:
        print("Error: Provided model is neither a .pt file to load or a FlexMLP class.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Prepare input
    inp = data[features].values if featureScaler is None else featureScaler.transform(data[features].values)
    inp = torch.from_numpy(inp.astype(np.float64))

    # Move everything to device
    model.to(device)
    inp.to(device)

    # Run model
    pred = model(inp)

    # move output to cpu
    pred.to("cpu")

    # Modify label name with suffix
    labels = [label+"_pred" for label in labels]

    # Rescale output
    df_pred = pd.DataFrame(pred, columns=labels) if labelScaler is None \
              else pd.DataFrame(labelScaler.inverse_transform(pred.detach().numpy()), columns=labels)

    return df_pred




