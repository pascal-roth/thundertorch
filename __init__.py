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

    import os
    import sys
    from sklearn.preprocessing import MinMaxScaler
    # Make sure right matplotlib backend is running
    import matplotlib
    import platform
    if platform.system() == "Darwin":
        matplotlib.use("MacOSX")
    else:
        matplotlib.use("qt5agg")
except ImportError as error:
    # Check if stfs_pytoolbox was installed with ML support
    print(error.__class__.__name__ + ": " + error.msg)
    print("Are you sure you have installed the stfs_pytoolbox with ML support?")
    print("Run Install_stfs_pytoolbox.sh again and answer with 'y' when asked for ML support!")


class FlexMLP(torch.nn.Module):
    """
    A Multi-Layer-Perceptron classes based on the torch.nn.Module, whos' architecture can be
    defined flexibly at object creation
    Examples
        model = FlexMLP(2, 5, [64,128,32])
        this returns a model with:
        2 input
        5 outputs
        3 hidden layer with 64, 128 and 32 neurons respectively
    """
    def __init__(self, n_inp, n_out, n_hidden_neurons=[32, 32], activation_fn=F.relu, output_activation=None):
        """
        Initializes a flexMLP model based on the provided parameters

        Parameters
        ----------
        n_inp               -   int number of input neurons
        n_out               -   int number of output neurons
        n_hidden_neurons    -   list of int  number of neurons per hidden layer (default=[32, 32])
        activation_fn       -   activation function for each hidden layer (default=F.relu)
        output_activation   -   activation function for output layer (default=None)
        """
        super().__init__()

        self.n_inp = n_inp
        self.n_out = n_out
        self.n_neurons = [n_inp] + n_hidden_neurons + [n_out]
        self.hidden_layers = len(n_hidden_neurons)
        self.layers = torch.nn.ModuleList()
        self.activation = activation_fn
        self.output_activation = output_activation

        # +1 to add last layer
        for i in range(len(self.n_neurons)-1):
            self.layers.append(torch.nn.Linear(self.n_neurons[i], self.n_neurons[i+1]))

    def forward(self, x):
        """
        Forward pass through model

        Parameters
        ----------
        x: torch.Tensor - model input

        Returns
        -------
        y: torch.Tensor - model output
        """

        # loop through all layer but last
        for i in range(self.hidden_layers):
            x = self.activation(self.layers[i](x))

        # output layer
        if(self.output_activation):
            x = self.output_activation(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
        return x


def createFlexMLPCheckpoint(model, path, features=None, labels=None, epochs=None, scalers=None):
    """
    function that saves FlexMLP model with model architecture and current parameters

    Parameters
    ----------
    model: FlexMLP
        model that will be saved
    path: str
        path to save location
    features: list str
        list of features names
    labels: list str
        list of label names
    epochs: int
        number of trained epochs
    scalers: list of sklearn.scalers
        feature- and labelscaler

    """

    checkpoint = {  'input_size': model.n_inp,
                    'output_size': model.n_out,
                    'hidden_layers': model.n_neurons[1:-1],  # cut of number of in and outputs
                    'activation': model.activation,
                    'output_activation': model.output_activation,
                    'state_dict': model.state_dict()}
    if epochs:
        checkpoint['epochs'] = epochs
    if features:
        checkpoint['features'] = features
    if labels:
        checkpoint['labels'] = labels
    if scalers:
        if not isinstance(scalers, list):
            print("Error you must provide list with feature and labelscaler. e.g. scalers=[featureScaler, labelScaler]")
            sys.exit(-1)
        checkpoint['scalers'] = scalers

    # Have to use dill as pickle module because softplus activation function is C wrapper and cannot
    # be saved by regular pickle
    import dill
    torch.save(checkpoint, path, pickle_module=dill)


def loadFlexMLPCheckpoint(filepath):
    """
    function that loads a flexMLP model from a file

    Parameters
    ----------
    filepath: str file name that holds FlexMLP checkpoint created with createFlexMLPCheckpoint(model, path)
    Returns
    -------
    model - FlexMLP object
    """
    # pyTorch saves device on which model was saved, therefore it has to be saved
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model = FlexMLP(checkpoint['input_size'],
                    checkpoint['output_size'],
                    checkpoint['hidden_layers'],
                    activation_fn=checkpoint['activation'],
                    output_activation=checkpoint['output_activation']).double()

    model.load_state_dict(checkpoint['state_dict'])

    if "epochs" in checkpoint:
        epochs = checkpoint["epochs"]
    if "features" in checkpoint:
        features = checkpoint["features"]
    if "labels" in checkpoint:
        labels = checkpoint["labels"]
    if "scalers" in checkpoint:
        scalers = checkpoint["scalers"]

    return model, features, labels, epochs, scalers


class AssemblyModel(torch.nn.Module):
    """
    model that combines various single torch models that have the same input but different output
    into one model for later convenience
    """
    def __init__(self,models,x_min, x_max, y_min, y_max, limit_scale=False):
        """

        Parameters
        ----------
        models: list torch.nn.modules   models that are to be combined
        x_min:  np.array oder torch.tensor  minimum value for input scaling
        x_max:  np.array oder torch.tensor  maximum value for input scaling
        y_min:  np.array oder torch.tensor  minimum value for output scaling
        y_max:  np.array oder torch.tensor  maximum value for output scaling
        """
        super().__init__()

        self.models = torch.nn.ModuleList(models)
        self.X_max = x_max if isinstance(x_max, torch.Tensor) else torch.tensor(x_max, dtype=torch.float64)
        self.X_min = x_min if isinstance(x_min, torch.Tensor) else torch.tensor(x_min, dtype=torch.float64)
        self.Y_max = y_max if isinstance(y_max, torch.Tensor) else torch.tensor(y_max, dtype=torch.float64)
        self.Y_min = y_min if isinstance(y_min, torch.Tensor) else torch.tensor(y_min, dtype=torch.float64)
        self.limit_scale = limit_scale

    def forward(self, Xorg):
        """
        Forward pass of model
            runs forward pass through all submodels and scales all in- and outputs

        Parameters
        ----------
        Xorg: torch.tensor  model input

        Returns
        -------
        Y: torch.tensor model output

        """
        X = Xorg.clone()
        X.requires_grad_(False)
        X = (X - self.X_min) / (self.X_max - self.X_min)
        # If input are out of range of trained scales, set value to border
        if self.limit_scale:
            X[X > 1] = 1
            X[X < 0] = 0
        outputs = []
        for i, model in enumerate(self.models):
            out = model(X)
            out = out * (self.Y_max[i] - self.Y_min[i]) + self.Y_min[i]
            # out = out.view(-1)
            outputs.append(out)
        return torch.cat(outputs, 1)

    def toTorchScript(self, path):
        """
        saves assembly model as torch-script for application in C++ Code

        Parameters
        ----------
        path: str   path + file name of model
        """
        n_inp = self.models[0].n_inp
        sample_input = torch.ones([8, n_inp], dtype=torch.float64)
        b = self.forward(sample_input)
        with torch.no_grad():
            torch_script = torch.jit.trace(self, sample_input)

        # Saving the model
        if(os.path.exists(path)):
            msg = "Model path already exists. Are you sure you want to overwrite existing model? [yes/no]"
            answer = ""
            while(answer != "yes" and answer != "no" and answer!= "y" and answer != "n"):
                answer = input(msg)
                if(answer == "no" or answer =="n"):
                    print("Execution aborted!")
                    return -1

        print("Saving assembly model as torchScript to {}".format(path))
        torch_script.save(path)


def trainFlexMLP(model, path, features, labels, df_train, df_validation=None, epochs=10,
                      batch=16, lr=0.001, loss_fn = torch.nn.MSELoss(), plot=False):
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

    # scale features with new min and max values and labels with all values
    featureScaler = MinMaxScaler()
    featureScaler.fit(np.stack((x_train_max, x_train_min), axis=0))
    labelScaler = MinMaxScaler()
    labelScaler.fit(y_train.values)



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

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # track losses, calculate initial validation loss and set that as baseline
    prediction = model.forward(x_validation_tensor)
    best_loss = loss_fn(prediction, y_validation_tensor).item()
    print("\nInitial validation loss is: {:6.5e}\n".format(best_loss))
    train_losses, validation_losses = [], []

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
        ax.set_ylim(1e-7, best_loss*10)
        plt.yscale("log")
        ax.plot(xdata, [best_loss], 'r-', label="Training loss")
        ax.plot(xdata, [best_loss], 'b-', label="Validation loss")
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
        inner.refresh() # flush last output
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

        # write output using progress bars
        outer.write("Epoch: {:05d}/{:05d}, Training loss: {:6.5e}, Validation loss: {:6.5e}".format(epoch, epochs, train_losses[-1], validation_losses[-1]))
        best_log.set_description_str("Best validation loss {:6.5e}".format(best_loss))

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

