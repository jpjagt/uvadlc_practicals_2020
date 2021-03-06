"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

activation_fns = {
    "ELU": nn.ELU,
    "RELU": nn.ReLU,
    "Hardshrink": nn.Hardshrink,
    "Tanh": nn.Tanh,
}


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, activation_fn="ELU"):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        super(MLP, self).__init__()

        if activation_fn not in activation_fns:
            raise "Invalid activation_fn"

        modules = []
        n_overall = [n_inputs] + n_hidden + [n_classes]
        # modules.append(nn.BatchNorm1d(n_inputs))

        for i in range(len(n_overall) - 1):
            modules.append(nn.Linear(n_overall[i], n_overall[i + 1]))
            if i < (len(n_overall) - 2):
                modules.append(nn.BatchNorm1d(n_overall[i + 1]))
                modules.append(activation_fns[activation_fn]())
        modules.append(nn.Softmax(dim=1))

        self.nn = nn.Sequential(*modules)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """
        out = self.nn(x.reshape(x.size()[0], -1))

        return out
