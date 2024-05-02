#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from eipl.utils import tensor2numpy, plt_img, get_feature_map


def create_position_encoding(width: int, height: int, normalized=True, data_format="channels_first"):
    if normalized:
        pos_x, pos_y = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="xy")
    else:
        pos_x, pos_y = np.meshgrid(
            np.linspace(0, height - 1, height),
            np.linspace(0, width - 1, width),
            indexing="xy",
        )

    if data_format == "channels_first":
        pos_xy = torch.from_numpy(np.stack([pos_x, pos_y], axis=0)).float()  # (2,W,H)
    else:
        pos_xy = torch.from_numpy(np.stack([pos_x, pos_y], axis=2)).float()  # (W,H,2)

    pos_x = torch.from_numpy(pos_x.reshape(height * width)).float()
    pos_y = torch.from_numpy(pos_y.reshape(height * width)).float()

    return pos_xy, pos_x, pos_y


class SpatialSoftmax(nn.Module):
    """Spatial Softmax
    Extract XY position from feature map of CNN

    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    ``Deep spatial autoencoders for visuomotor learning.``
    2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016.
    https://ieeexplore.ieee.org/abstract/document/7487173
    """

    def __init__(self, width: int, height: int, temperature=1e-4, normalized=True):
        super(SpatialSoftmax, self).__init__()
        self.width = width
        self.height = height
        self.temperature = temperature

        _, pos_x, pos_y = create_position_encoding(width, height, normalized=normalized)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        assert height == self.height
        assert width == self.width

        # flatten, apply softmax
        logit = x.reshape(batch_size, channels, -1)
        att_map = torch.softmax(logit / self.temperature, dim=-1)

        # compute expectation
        expected_x = torch.sum(self.pos_x * att_map, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * att_map, dim=-1, keepdim=True)
        keys = torch.cat([expected_x, expected_y], -1)

        # keys [[x,y], [x,y], [x,y],...]
        keys = keys.reshape(batch_size, channels, 2)
        att_map = att_map.reshape(-1, channels, width, height)
        return keys, att_map


class InverseSpatialSoftmax(nn.Module):
    """InverseSpatialSoftmax
    Generate heatmap from XY position

    Hideyuki Ichiwara, Hiroshi Ito, Kenjiro Yamamoto, Hiroki Mori, Tetsuya Ogata
    ``Spatial Attention Point Network for Deep-learning-based Robust Autonomous Robot Motion Generation.``
    https://arxiv.org/abs/2103.01598
    """

    def __init__(self, width: int, height: int, heatmap_size=0.1, normalized=True, convex=True):
        super(InverseSpatialSoftmax, self).__init__()

        self.width = width
        self.height = height
        self.normalized = normalized
        self.heatmap_size = heatmap_size
        self.convex = convex

        pos_xy, _, _ = create_position_encoding(width, height, normalized=normalized)
        self.register_buffer("pos_xy", pos_xy)

    def forward(self, keys):
        squared_distances = torch.sum(torch.pow(self.pos_xy[None, None] - keys[:, :, :, None, None], 2.0), axis=2)
        heatmap = torch.exp(-squared_distances / self.heatmap_size)

        if self.convex:
            heatmap = torch.abs(1.0 - heatmap)

        return heatmap
