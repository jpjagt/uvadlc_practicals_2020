# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_length,
        vocabulary_size,
        lstm_num_hidden=256,
        lstm_num_layers=2,
        device="cuda:0",
    ):

        super(TextGenerationModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.lstm = nn.LSTM(
            input_size=vocabulary_size,
            hidden_size=lstm_num_hidden,
            batch_first=True,
            num_layers=lstm_num_layers,
        ).to(device)
        self.Wph = nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty((lstm_num_hidden, vocabulary_size))
            )
        ).to(device)
        self.bp = nn.Parameter(torch.zeros((vocabulary_size,))).to(device)

    def forward(self, x, init_hidden_and_state=None, temperature=1.0):
        x_one_hot = nn.functional.one_hot(x, self.vocabulary_size).float()
        lstm_out, hidden_and_state_out = self.lstm(
            x_one_hot, init_hidden_and_state
        )
        self.hidden_and_state_out = hidden_and_state_out

        out = (lstm_out @ self.Wph) + self.bp
        return out
        # return torch.nn.functional.log_softmax(out, dim=1)
