"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict

import torch.nn as nn
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class PreAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
    ):
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        return x + self.net(x)


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """
        super(ConvNet, self).__init__()

        self.nn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            n_channels,
                            64,
                            kernel_size=(3, 3),
                            stride=1,
                            padding=1,
                        ),
                    ),
                    (
                        "PreAct1",
                        PreAct2d(64, 64),
                    ),
                    (
                        "conv1",
                        nn.Conv2d(
                            64, 128, kernel_size=(1, 1), stride=1, padding=0
                        ),
                    ),
                    (
                        "maxpool1",
                        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
                    ),
                    (
                        "PreAct2_a",
                        PreAct2d(128, 128),
                    ),
                    (
                        "PreAct2_b",
                        PreAct2d(128, 128),
                    ),
                    (
                        "conv2",
                        nn.Conv2d(
                            128, 256, kernel_size=(1, 1), stride=1, padding=0
                        ),
                    ),
                    (
                        "maxpool2",
                        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
                    ),
                    (
                        "PreAct3_a",
                        PreAct2d(256, 256),
                    ),
                    (
                        "PreAct3_b",
                        PreAct2d(256, 256),
                    ),
                    (
                        "conv3",
                        nn.Conv2d(
                            256, 512, kernel_size=(1, 1), stride=1, padding=0
                        ),
                    ),
                    (
                        "maxpool3",
                        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
                    ),
                    (
                        "PreAct4_a",
                        PreAct2d(512, 512),
                    ),
                    (
                        "PreAct4_b",
                        PreAct2d(512, 512),
                    ),
                    (
                        "maxpool4",
                        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
                    ),
                    (
                        "PreAct5_a",
                        PreAct2d(512, 512),
                    ),
                    (
                        "PreAct5_b",
                        PreAct2d(512, 512),
                    ),
                    (
                        "maxpool5",
                        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
                    ),
                    ("flatten", Flatten()),
                    ("linear", nn.Linear(512, n_classes)),
                ]
            )
        )

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

        out = self.nn(x)
        return out
