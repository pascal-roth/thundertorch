import torch
import os
from typing import List, Optional, Union

from thunder_torch.models.LightningFlexMLP import LightningFlexMLP


class AssemblyModel(torch.nn.Module):
    """
    model that combines various single torch models that have the same input but different output
    into one model for later convenience
    """
    def __init__(self, models: List[LightningFlexMLP],
                 x_min: Optional[Union[List[float], torch.Tensor]] = None,
                 x_max: Optional[Union[List[float], torch.Tensor]] = None,
                 y_min: Optional[Union[List[float], torch.Tensor]] = None,
                 y_max: Optional[Union[List[float], torch.Tensor]] = None,
                 limit_scale: bool = False) -> None:
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
        if all(elem is not None for elem in [x_max, x_min, y_max, y_min]):
            X_max = x_max if isinstance(x_max, torch.Tensor) else torch.tensor(x_max, dtype=torch.float64)
            X_min = x_min if isinstance(x_min, torch.Tensor) else torch.tensor(x_min, dtype=torch.float64)
            Y_max = y_max if isinstance(y_max, torch.Tensor) else torch.tensor(y_max, dtype=torch.float64)
            Y_min = y_min if isinstance(y_min, torch.Tensor) else torch.tensor(y_min, dtype=torch.float64)

            # register scaling parameters as buffers that their device will also be changed then
            # calling .to .cuda or .cpu
            self.register_buffer("X_max", X_max)
            self.register_buffer("X_min", X_min)
            self.register_buffer("Y_max", Y_max)
            self.register_buffer("Y_min", Y_min)

            self.X_max: torch.Tensor
            self.X_min: torch.Tensor
            self.Y_max: torch.Tensor
            self.Y_min: torch.Tensor

            self.min_max_scale = True
        else:
            self.min_max_scale = False

        self.limit_scale = limit_scale

        # Save features and labels as attributes of the torch script, emtpy list must be typed for the tracing
        self.labels = torch.jit.annotate(List[str], [])
        for model in models:
            self.labels.append(model.hparams.lparams.labels)
        self.features = models[0].hparams.lparams.features

    def forward(self, Xorg: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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

        if self.min_max_scale:
            X = (X - self.X_min) / (self.X_max - self.X_min)
        # If input are out of range of trained scales, set value to border
        # if self.limit_scale:
        #     X[X > 1] = 1
        #     X[X < 0] = 0
        outputs = []

        # enumerate leads to errors since ModuleList not Iterable
        for i in range(len(self.models)):
            out = self.models[i](X)

            if self.min_max_scale:
                out = out * (self.Y_max[i] - self.Y_min[i]) + self.Y_min[i]

            # out = out.view(-1)
            outputs.append(out)
        return torch.cat(outputs, 1)

    # Currently this function can only be traced using torch.jit.trace_module
    # but not exported using torch.jit.script(self)
    # For some reason self.models are unknown and cannot be iterated over, please try to use it in a later torch version
    # as of now torch 1.7 and earlier does not work
    # #@torch.jit.export
    def forward_parallel(self, Xorg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of model
            runs forward pass through all submodels in parallel using torch::jit::fork and scales all in- and outputs
            for details see: https://pytorch.org/tutorials/advanced/torch-script-parallelism.html
        Parameters
        ----------
        Xorg: torch.tensor  model input

        Returns
        -------
        Y: torch.tensor model output

        """
        X = Xorg.clone()
        X.requires_grad_(False)

        if self.min_max_scale:
            X = (X - self.X_min) / (self.X_max - self.X_min)

        # If input are out of range of trained scales, set value to border
        if self.limit_scale:
            X[X > 1] = 1
            X[X < 0] = 0

        # jit.fork spawns an asynchronous task that are run in parallel until jit.wait
        # in pt 1.2 fork and wait are called with leading underscore
        # mypy - ModuleList has no attribute __next__, however working
        futures = [torch.jit._fork(model, X) for model in self.models]  # type: ignore[attr-defined]
        outputs = [torch.jit._wait(fut) for fut in futures]
        return torch.cat(outputs, 1)

    def toTorchScript(self, path: str) -> None:
        # TODO introduce general export scripts in utils to let every model be exportable
        """
        saves assembly model as torch-script for application in C++ Code

        Parameters
        ----------
        path: str   path + file name of model
        """
        # mypy does not recognize LightningFlexMLP and its Namespace correctly, see again after change of function
        n_inp: int = self.models[0].hparams.n_inp  # type: ignore
        sample_input = torch.ones([8, n_inp], dtype=torch.float64)
        # b = self.forward(sample_input)
        with torch.no_grad():
            # we have to use the trace_module function here to trace multiple functions besides forward
            torch_script = torch.jit.trace_module(self, {"forward": sample_input, "forward_parallel": sample_input})

        # Saving the model
        if os.path.exists(path):
            msg = "Model path already exists. Are you sure you want to overwrite existing model? [yes/no]"
            answer = ""
            while answer != "yes" and answer != "no" and answer != "y" and answer != "n":
                answer = input(msg)
                if answer == "no" or answer == "n":
                    print("Execution aborted!")
                    return

        print("Saving assembly model as torchScript to {}".format(path))
        torch_script.save(path)

    def to_onnx(self, path: str) -> None:
        """
        Function to save assembly model in onnx format
        Parameters
        ----------
        path:   str
            path where .onnx file is saved
        dtype:  torch.dtype, default: torch.float
            dtype of saved model
        """
        import torch.onnx
        # mypy does not recognize LightningFlexMLP and its Namespace correctly, see again after change of function
        n_inp: int = self.models[0].hparams.n_inp  # type: ignore
        dtype = self.models[0].dtype
        x = torch.ones([8, n_inp], dtype=dtype)  # type: ignore

        # Export the model
        torch.onnx.export(self,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=9,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
