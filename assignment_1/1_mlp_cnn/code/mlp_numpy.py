"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
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
          neg_slope: negative slope parameter for LeakyReLU

        TODO:
        Implement initialization of the network.
        """

        self.modules = []

        n_overall = [n_inputs] + n_hidden + [n_classes]
        for i in range(len(n_overall) - 1):
            self.modules.append(LinearModule(n_overall[i], n_overall[i + 1]))
            if i < (len(n_overall) - 2):
                self.modules.append(ELUModule())
        self.modules.append(SoftMaxModule())

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

        out = x.reshape(x.shape[0], -1)
        for module in self.modules:
            out = module.forward(out)

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        for module in self.modules[::-1]:
            dout = module.backward(dout)
