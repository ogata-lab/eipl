#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.utils import get_activation_fn
from eipl.layer import MTRNNCell


class BasicLSTM(nn.Module):
    #:: BasicLSTM
    """BasicLSTM

    Arguments:
        in_dim (int):  Number of fast context neurons
        rec_dim (int): Number of fast context neurons
        out_dim (int): Number of fast context neurons
        activation (string, optional): If you set `None`, no activation is applied (ie. "linear" activation: `a(x) = x`).
            The default is hyperbolic tangent (`tanh`).
    """

    def __init__(self, in_dim, rec_dim, out_dim, activation="tanh"):
        super(BasicLSTM, self).__init__()

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = get_activation_fn(activation)

        self.rnn = nn.LSTMCell(in_dim, rec_dim)
        self.rnn_out = nn.Sequential(nn.Linear(rec_dim, out_dim), activation)

    def forward(self, x, state=None):
        rnn_hid = self.rnn(x, state)
        y_hat = self.rnn_out(rnn_hid[0])

        return y_hat, rnn_hid


class BasicMTRNN(nn.Module):
    #:: BasicMTRNN
    """BasicMTRNN

    Arguments:
        in_dim (int):  Number of fast context neurons
        rec_dim (int): Number of fast context neurons
        out_dim (int): Number of fast context neurons
        activation (string, optional): If you set `None`, no activation is applied (ie. "linear" activation: `a(x) = x`).
            The default is hyperbolic tangent (`tanh`).
    """

    def __init__(
        self, in_dim, fast_dim, slow_dim, fast_tau, slow_tau, out_dim=None, activation="tanh"
    ):
        super(BasicMTRNN, self).__init__()

        if out_dim is None:
            out_dim = in_dim

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = get_activation_fn(activation)

        self.mtrnn = MTRNNCell(
            in_dim, fast_dim, slow_dim, fast_tau, slow_tau, activation=activation
        )
        # Output of RNN
        self.rnn_out = nn.Sequential(nn.Linear(fast_dim, out_dim), activation)

    def forward(self, x, state=None):
        rnn_hid = self.mtrnn(x, state)
        y_hat = self.rnn_out(rnn_hid[0])

        return y_hat, rnn_hid
