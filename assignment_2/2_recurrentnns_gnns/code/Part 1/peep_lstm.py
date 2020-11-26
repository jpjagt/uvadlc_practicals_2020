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
    param = torch.empty(*size).float()
    return nn.Parameter(getattr(nn.init, init)(param))


class LSTMInternalLayer(nn.Module):
    def __init__(self, dim_x, dim_h, dim_out, activation_fn):
        super(LSTMInternalLayer, self).__init__()

        self.weights_x = init_param(size=(dim_out, dim_x))
        self.weights_h = init_param(size=(dim_out, dim_h))
        self.bias = init_param(size=(dim_out,), init="zeros_")
        self.activation_fn = activation_fn

    def forward(self, x, h):
        return self.activation_fn(
            self.weights_x @ x.T + self.weights_h @ h + self.bias
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
        emb_dim=6,
        num_input_digits=3,
    ):

        super(peepLSTM, self).__init__()

        self.__num_input_digits = num_input_digits

        self.__dim_h = hidden_dim
        self.__dim_out = hidden_dim
        self.__batch_size = batch_size

        self.emb_dim = emb_dim
        self.__dim_x = self.emb_dim

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

        self.weights_p = init_param(size=(num_classes, self.__dim_h))
        self.bias_p = init_param(size=(num_classes,), init="zeros_")

        self.weights_x2c = init_param(size=(self.__dim_h, self.__dim_x))
        self.bias_x2c = init_param(size=(self.__dim_h,), init="zeros_")

        self.embedding = nn.Embedding(
            num_embeddings=num_input_digits, embedding_dim=self.emb_dim
        )

        self.to(device)

    def forward(self, x):
        # h = torch.zeros(self.__batch_size, self.__dim_h)
        prev_c = torch.zeros(self.__batch_size, self.__dim_h)

        x = self.embedding(x.long().squeeze())

        for t in range(x.size()[1]):
            x_t = x[:, t]

            f = self.f(x_t, prev_c)
            i = self.i(x_t, prev_c)
            o = self.o(x_t, prev_c)
            c = (
                torch.sigmoid(self.weights_x2c @ x_t.T + self.bias_x2c) * i
                + prev_c * f
            )
            h = torch.tanh(c) * o

            prev_c = c
        p_t = (self.weights_p @ h).T + self.bias_p
        # not doing softmax because i'm using CrossEntropyLoss
        # and it does not change the accuracy measurement
        return p_t
