# import packages
import torch
from typing import List, Union, Tuple, Optional


class LayerConstructors:
    def __init__(self):
        self.final_channel: int

        self.layers_list: List = []
        self.layers: torch.nn.Sequential
        self.height: int
        self.width: int
        self.depth: int
        self.layer_activation = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear,
                                 torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,)

    def construct_nn2d(self, layer_list: list) -> None:
        """
        Functionality to build any kind of torch.nn 2d layer (convolutional, pooling, padding, normalization, recurrent,
        dropout, linear ...)

        Parameters
        ----------
        layer_list          - list of layers with the corresponding parameters
        """
        # Construct all conv layers
        for layer_dict in layer_list:
            if 'params' in layer_dict:
                activation = layer_dict.get('activation', True)
                self.layers_list.append(getattr(torch.nn, layer_dict['type'])(**layer_dict['params']))
            else:
                activation = True
                self.layers_list.append(getattr(torch.nn, layer_dict['type'])())

            if all(hasattr(self.layers_list[-1], elem) for elem in ['padding', 'stride', 'kernel_size']):
                if isinstance(self.layers_list[-1].padding, tuple):
                    self.height = int((self.height + 2 * self.layers_list[-1].padding[0]) /
                                      self.layers_list[-1].stride[0]) - (self.layers_list[-1].kernel_size[0] - 1)
                    self.width = int((self.width + 2 * self.layers_list[-1].padding[1]) /
                                     self.layers_list[-1].stride[1]) - (self.layers_list[-1].kernel_size[1] - 1)
                else:
                    self.height = int((self.height + 2 * self.layers_list[-1].padding) /
                                      self.layers_list[-1].stride) - (self.height % self.layers_list[-1].kernel_size)
                    self.width = int((self.width + 2 * self.layers_list[-1].padding) /
                                     self.layers_list[-1].stride) - (self.width % self.layers_list[-1].kernel_size)

            if isinstance(self.layers_list[-1], self.layer_activation) and activation:
                self.layers_list.append(self.activation_fn)

    def construct_nn3d(self, layer_list: list) -> None:
        """
        Functionality to build any kind of torch.nn 3d layer (convolutional, pooling, padding, normalization, recurrent,
        dropout, linear ...)

        Parameters
        ----------
        layer_list          - list of layers with the corresponding parameters
        """
        # Construct all conv layers
        for layer_dict in layer_list:
            if 'params' in layer_dict:
                activation = layer_dict.get('activation', True)
                self.layers_list.append(getattr(torch.nn, layer_dict['type'])(**layer_dict['params']))
            else:
                activation = True
                self.layers_list.append(getattr(torch.nn, layer_dict['type'])())

            if all(hasattr(self.layers_list[-1], elem) for elem in ['padding', 'stride', 'kernel_size']):
                if isinstance(self.layers_list[-1].padding, tuple):
                    self.height = int((self.height + 2 * self.layers_list[-1].padding[0]) /
                                      self.layers_list[-1].stride[0]) - (self.layers_list[-1].kernel_size[0] - 1)
                    self.width = int((self.width + 2 * self.layers_list[-1].padding[1]) /
                                     self.layers_list[-1].stride[1]) - (self.layers_list[-1].kernel_size[1] - 1)
                    self.depth = int((self.depth + 2 * self.layers_list[-1].padding[2]) /
                                     self.layers_list[-1].stride[2]) - (self.layers_list[-1].kernel_size[2] - 1)
                else:
                    self.height = int((self.height + 2 * self.layers_list[-1].padding) /
                                      self.layers_list[-1].stride) - (self.height % self.layers_list[-1].kernel_size)
                    self.width = int((self.width + 2 * self.layers_list[-1].padding) /
                                     self.layers_list[-1].stride) - (self.width % self.layers_list[-1].kernel_size)
                    self.depth = int((self.depth + 2 * self.layers_list[-1].padding) /
                                     self.layers_list[-1].stride) - (self.width % self.layers_list[-1].kernel_size)

            if isinstance(self.layers_list[-1], self.layer_activation) and activation:
                self.layers_list.append(self.activation_fn)

    def construct_mlp(self, in_dim: int, hidden_layer: List[int], out_dim: int) -> None:
        """
        Quick functionality to built a MLP network

        Parameters
        ----------
        in_dim              - input dimensions
        hidden_layer        - dimensions of the hidden layers
        out_dim             - output dimensions
        """
        # TODO: think about adding a bias flag in case a linear function should be learned
        if hidden_layer:
            # Construct all MLP layers
            self.layers_list.append(torch.nn.Linear(in_dim, hidden_layer[0]))
            self.layers_list.append(self.activation_fn)

            layer_sizes = zip(hidden_layer[:-1], hidden_layer[1:])

            for h1, h2 in layer_sizes:
                self.layers_list.append(torch.nn.Linear(h1, h2))
                self.layers_list.append(self.activation_fn)

            self.layers_list.append(torch.nn.Linear(hidden_layer[-1], out_dim))
        else:
            # in the case no hidden layers are given
            self.layers_list.append(torch.nn.Linear(in_dim, out_dim))
