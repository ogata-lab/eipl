#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch.nn as nn


def get_activation_fn(name, inplace=True):
    if name.casefold() == "relu":
        return nn.ReLU(inplace=inplace)
    elif name.casefold() == "lrelu":
        return nn.LeakyReLU(inplace=inplace)
    elif name.casefold() == "softmax":
        return nn.Softmax()
    elif name.casefold() == "tanh":
        return nn.Tanh()
    else:
        assert False, "Unknown activation function {}".format(name)
