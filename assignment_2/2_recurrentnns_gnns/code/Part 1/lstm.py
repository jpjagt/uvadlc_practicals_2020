"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_param(size, init="kaiming_normal_"):
    param = torch.empty(*size)
    return nn.Parameter(getattr(nn.init, init)(param))


class LSTMInternalLayer(nn.Module):
    def __init__(self, dim_x, dim_h, dim_out, activation_fn):
        super(LSTMInternalLayer, self).__init__()

        self.weights_x = init_param(size=(dim_out, dim_x)).T
        self.weights_h = init_param(size=(dim_out, dim_h)).T
        self.bias = init_param(size=(dim_out,), init="zeros_")
        self.activation_fn = activation_fn

    def forward(self, x, h):
        return self.activation_fn(
            torch.matmul(x, self.weights_x)
            + torch.matmul(h, self.weights_h)
            + self.bias
        )


class LSTM(nn.Module):
    def __init__(
        self,
        seq_length,
        input_dim,
        hidden_dim,
        num_classes,
        batch_size,
        device,
        num_input_digits=10,
    ):

        super(LSTM, self).__init__()

        self.__num_input_digits = num_input_digits

        self.__dim_x = self.__num_input_digits
        self.__dim_h = hidden_dim
        self.__dim_out = hidden_dim
        self.__batch_size = batch_size

        self.g = LSTMInternalLayer(
            self.__dim_x,
            self.__dim_h,
            self.__dim_out,
            activation_fn=torch.tanh,
        )
        self.i = LSTMInternalLayer(
            self.__dim_x,
            self.__dim_h,
            self.__dim_out,
            activation_fn=torch.sigmoid,
        )
        self.f = LSTMInternalLayer(
            self.__dim_x,
            self.__dim_h,
            self.__dim_out,
            activation_fn=torch.sigmoid,
        )
        self.o = LSTMInternalLayer(
            self.__dim_x,
            self.__dim_h,
            self.__dim_out,
            activation_fn=torch.sigmoid,
        )

        self.weights_p = init_param(size=(num_classes, self.__dim_h)).T
        self.bias_p = init_param(size=(num_classes,), init="zeros_")

        self.to(device)

    def to_onehot(self, x):
        return F.one_hot(x, self.__num_input_digits).float()

    def forward(self, x):
        prev_h = torch.zeros(self.__batch_size, self.__dim_h)
        prev_c = torch.zeros(self.__batch_size, self.__dim_h)

        x = x.long().squeeze()

        for t in range(x.size()[1]):
            x_t_embedded = self.to_onehot(x[:, t].squeeze())

            g = self.g(x_t_embedded, prev_h)
            i = self.i(x_t_embedded, prev_h)
            f = self.f(x_t_embedded, prev_h)
            o = self.o(x_t_embedded, prev_h)
            c = g * i + prev_c * f
            h = torch.tanh(c) * o

            prev_h = h
            prev_c = c

            p = torch.matmul(h, self.weights_p) + self.bias_p
        return F.log_softmax(p, dim=1)
