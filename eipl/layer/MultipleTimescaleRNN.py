#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.utils import get_activation_fn


class MTRNNCell(nn.Module):
    #:: MTRNNCell
    """Multiple Timescale RNN.

    Implements a form of Recurrent Neural Network (RNN) that operates with multiple timescales.
    This is based on the idea of hierarchical organization in human cognitive functions.

    Arguments:
        input_dim (int): Number of input features.
        fast_dim (int): Number of fast context neurons.
        slow_dim (int): Number of slow context neurons.
        fast_tau (float): Time constant value of fast context.
        slow_tau (float): Time constant value of slow context.
        activation (string, optional): If you set `None`, no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias (Boolean, optional): whether the layer uses a bias vector. The default is False.
        use_pb (Boolean, optional): whether the recurrent uses a pb vector. The default is False.

    Yuichi Yamashita, Jun Tani,
    "Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment.", NeurIPS 2018.
    https://arxiv.org/abs/1807.03247v2
    """

    def __init__(
        self,
        input_dim,
        fast_dim,
        slow_dim,
        fast_tau,
        slow_tau,
        activation="tanh",
        use_bias=False,
        use_pb=False,
    ):
        super(MTRNNCell, self).__init__()

        self.input_dim = input_dim
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.fast_tau = fast_tau
        self.slow_tau = slow_tau
        self.use_bias = use_bias
        self.use_pb = use_pb

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

        # Input Layers
        self.i2f = nn.Linear(input_dim, fast_dim, bias=use_bias)

        # Fast context layer
        self.f2f = nn.Linear(fast_dim, fast_dim, bias=False)
        self.f2s = nn.Linear(fast_dim, slow_dim, bias=use_bias)

        # Slow context layer
        self.s2s = nn.Linear(slow_dim, slow_dim, bias=False)
        self.s2f = nn.Linear(slow_dim, fast_dim, bias=use_bias)

    def forward(self, x, state=None, pb=None):
        """Forward propagation of the MTRNN.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            state (list): Previous states (h_fast, h_slow, u_fast, u_slow), each of shape (batch_size, context_dim).
                   If None, initialize states to zeros.
            pb (bool): pb vector. Used if self.use_pb is set to True.

        Returns:
            new_h_fast (torch.Tensor): Updated fast context state.
            new_h_slow (torch.Tensor): Updated slow context state.
            new_u_fast (torch.Tensor): Updated fast internal state.
            new_u_slow (torch.Tensor): Updated slow internal state.
        """
        batch_size = x.shape[0]
        if state is not None:
            prev_h_fast, prev_h_slow, prev_u_fast, prev_u_slow = state
        else:
            device = x.device
            prev_h_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            prev_h_slow = torch.zeros(batch_size, self.slow_dim).to(device)
            prev_u_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            prev_u_slow = torch.zeros(batch_size, self.slow_dim).to(device)

        new_u_fast = (1.0 - 1.0 / self.fast_tau) * prev_u_fast + 1.0 / self.fast_tau * (
            self.i2f(x) + self.f2f(prev_h_fast) + self.s2f(prev_h_slow)
        )

        _input_slow = self.f2s(prev_h_fast) + self.s2s(prev_h_slow)
        if pb is not None:
            _input_slow += pb

        new_u_slow = (
            1.0 - 1.0 / self.slow_tau
        ) * prev_u_slow + 1.0 / self.slow_tau * _input_slow

        new_h_fast = self.activation(new_u_fast)
        new_h_slow = self.activation(new_u_slow)

        return new_h_fast, new_h_slow, new_u_fast, new_u_slow
