"""
This module implements a LSTM with peephole connections in PyTorch.
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


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()

        self.weights = init_param(size=(dim_out, dim_in)).T
        self.bias = init_param(size=(dim_out,), init="zeros_")

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias


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


class peepLSTM(nn.Module):
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

        super(peepLSTM, self).__init__()

        self.__num_input_digits = num_input_digits

        self.__dim_x = self.__num_input_digits
        self.__dim_h = hidden_dim
        self.__dim_out = hidden_dim
        self.__batch_size = batch_size

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

        self.linear_x2c = Linear(self.__dim_x, self.__dim_h)
        self.linear_p = Linear(num_classes, self.__dim_h)

        self.to(device)

    def to_onehot(self, x):
        return F.one_hot(x, self.__num_input_digits).float()

    def forward(self, x):
        prev_h = torch.zeros(self.__batch_size, self.__dim_h)
        prev_c = torch.zeros(self.__batch_size, self.__dim_h)

        x = x.long().squeeze()

        for t in range(x.size()[1]):
            x_t_embedded = self.to_onehot(x[:, t].squeeze())

            i = self.i(x_t_embedded, prev_c)
            f = self.f(x_t_embedded, prev_c)
            o = self.o(x_t_embedded, prev_c)
            c = torch.sigmoid(self.linear_x2c(x)) * i + prev_c * f
            h = torch.tanh(c) * o

            prev_c = c

            p = self.linear_p(h)
        return F.log_softmax(p, dim=1)
