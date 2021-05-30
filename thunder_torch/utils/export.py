# import torch
# import os
# import pytorch_lightning as pl
#
# from typing import Union, List
#
# from thunder_torch.utils import dynamic_imp
# from thunder_torch.models import LightningFlexMLP
#
#
# def toTorchScript(model: Union[torch.nn.ModuleList, pl.LightningModule, str, os.PathLike],
#                   input_size: Union[List[int], int],
#                   path: Union[str, os.PathLike]) -> None:
#     # TODO introduce general export scripts in utils to let every model be exportable
#     """
#     saves assembly model as torch-script for application in C++ Code
#
#     Parameters
#     ----------
#     path: str   path + file name of model
#     """
#     if isinstance(model, str) or isinstance(model, os.PathLike):
#         pl_model = dynamic_imp(model)
#     else:
#         pl_model = model
#
#     if type(pl_model) ==
#     # mypy does not recognize LightningFlexMLP and its Namespace correctly, see again after change of function
#     n_inp: int = self.models[0].hparams.n_inp  # type: ignore
#     sample_input = torch.ones([8, n_inp], dtype=torch.float64)
#     # b = self.forward(sample_input)
#     with torch.no_grad():
#         # we have to use the trace_module function here to trace multiple functions besides forward
#         torch_script = torch.jit.trace_module(model, {"forward": sample_input, "forward_parallel": sample_input})
#
#     # Saving the model
#     if os.path.exists(path):
#         msg = "Model path already exists. Are you sure you want to overwrite existing model? [yes/no]"
#         answer = ""
#         while answer != "yes" and answer != "no" and answer != "y" and answer != "n":
#             answer = input(msg)
#             if answer == "no" or answer == "n":
#                 print("Execution aborted!")
#                 return
#
#     print("Saving assembly model as torchScript to {}".format(path))
#     torch_script.save(path)
#
#
# def to_onnx(model, input_size, path: str) -> None:
#     """
#     Function to save assembly model in onnx format
#     Parameters
#     ----------
#     path:   str
#         path where .onnx file is saved
#     dtype:  torch.dtype, default: torch.float
#         dtype of saved model
#     """
#     import torch.onnx
#     # mypy does not recognize LightningFlexMLP and its Namespace correctly, see again after change of function
#     n_inp: int = self.models[0].hparams.n_inp  # type: ignore
#     dtype = self.models[0].dtype
#     x = torch.ones([8, n_inp], dtype=dtype)  # type: ignore
#
#     # Export the model
#     torch.onnx.export(self,  # model being run
#                       x,  # model input (or a tuple for multiple inputs)
#                       path,  # where to save the model (can be a file or file-like object)
#                       export_params=True,  # store the trained parameter weights inside the model file
#                       opset_version=9,  # the ONNX version to export the model to
#                       do_constant_folding=True,  # whether to execute constant folding for optimization
#                       input_names=['input'],  # the model's input names
#                       output_names=['output'],  # the model's output names
#                       dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#                                     'output': {0: 'batch_size'}})
