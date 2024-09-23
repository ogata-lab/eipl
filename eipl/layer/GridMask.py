#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import numpy as np


class GridMask:
    """GridMask

    Arguments:
        p (flaot, optional): Mask spacing
        d_range (Boolean, optional):
        r (Boolean, optional): This parameter determines how much of the original image is retained.
                                If r=0, the entire image is masked; if r=1, the image is not masked at all.

    Chen, Pengguang, et al. "Gridmask data augmentation."
    https://arxiv.org/abs/2001.04086
    """

    def __init__(self, p=0.6, d_range=(10, 30), r=0.6, channel_first=True):
        self.p = p
        self.d_range = d_range
        self.r = r
        self.channel_first = channel_first

    def __call__(self, img, debug=False):
        if not debug and np.random.uniform() > self.p:
            return img

        side = img.shape[-2]
        d = np.random.randint(*self.d_range, dtype=np.uint8)
        r = int(self.r * d)

        mask = np.ones((side + d, side + d), dtype=np.uint8)
        for i in range(0, side + d, d):
            for j in range(0, side + d, d):
                mask[i : i + (d - r), j : j + (d - r)] = 0

        delta_x, delta_y = np.random.randint(0, d, size=2)
        mask = mask[delta_x : delta_x + side, delta_y : delta_y + side]

        if self.channel_first:
            img *= np.expand_dims(mask, 0)
        else:
            img *= np.expand_dims(mask, -1)

        return img
