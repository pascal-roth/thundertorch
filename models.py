import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


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
    if 'activation' not in checkpoint:
        checkpoint['activation']=F.relu

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
    def __init__(self, models, x_min, x_max, y_min, y_max, limit_scale=False):
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
        if os.path.exists(path):
            msg = "Model path already exists. Are you sure you want to overwrite existing model? [yes/no]"
            answer = ""
            while answer != "yes" and answer != "no" and answer != "y" and answer != "n":
                answer = input(msg)
                if answer == "no" or answer == "n":
                    print("Execution aborted!")
                    return -1

        print("Saving assembly model as torchScript to {}".format(path))
        torch_script.save(path)
