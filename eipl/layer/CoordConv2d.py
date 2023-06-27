#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
import torch.nn.modules.conv as conv


class AddCoords(nn.Module):
    """AddCoords

    Arguments:
        min_range (flaot, optional): Minimum value for xx_channel, yy_channel. The default is 0. But original paper is -1.
        with_r (Boolean, optional): Wether add the radial channel. The default is True.
    """

    def __init__(self, min_range=0.0, with_r=False):
        super(AddCoords, self).__init__()
        self.min_range = min_range
        self.with_r = with_r

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        device = x.device

        xx_channel, yy_channel = torch.meshgrid(
            torch.linspace(self.min_range, 1.0, height, dtype=torch.float32),
            torch.linspace(self.min_range, 1.0, width, dtype=torch.float32),
            indexing="ij",
        )
        xx_channel = xx_channel.expand(batch_size, 1, width, height).to(device)
        yy_channel = yy_channel.expand(batch_size, 1, width, height).to(device)

        y = torch.cat([x, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            y = torch.cat([y, rr], dim=1)

        return y


class CoordConv2d(conv.Conv2d):
    """CoordConv2d
    Rosanne Liu, Joel Lehman, Piero Molino, Felipe Petroski Such, Eric Frank, Alex Sergeev, Jason Yosinski,
    ``An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution.``,
    NeurIPS 2018.
    https://arxiv.org/abs/1807.03247v2
    """

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        min_range=0.0,
        with_r=False,
    ):
        super(CoordConv2d, self).__init__(
            input_size, output_size, kernel_size, stride, padding, dilation, groups, bias
        )
        rank = 2
        self.addcoords = AddCoords(min_range=min_range, with_r=with_r)
        self.conv = nn.Conv2d(
            input_size + rank + int(with_r),
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        hid = self.addcoords(x)
        y = self.conv(hid)

        return y
