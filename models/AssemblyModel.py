import torch
import os


class AssemblyModel(torch.nn.Module):
    """
    model that combines various single torch models that have the same input but different output
    into one model for later convenience
    """
    def __init__(self, models: list, x_min, x_max, y_min, y_max, limit_scale: bool = False) -> None:
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

    def forward(self, Xorg: torch.tensor):
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

    def toTorchScript(self, path: str) -> None:
        """
        saves assembly model as torch-script for application in C++ Code

        Parameters
        ----------
        path: str   path + file name of model
        """
        n_inp = self.models[0].hparams.n_inp
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