#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.layer import CoordConv2d, AddCoords


class BasicCAE(nn.Module):
    #:: BasicCAE
    """BasicCAE"""

    def __init__(self, feat_dim=10):
        super(BasicCAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, 2, 1),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 2, 1),
            nn.Tanh(),
            nn.Conv2d(16, 12, 3, 2, 1),
            nn.Tanh(),
            nn.Conv2d(12, 8, 3, 2, 1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 50),
            nn.Tanh(),
            nn.Linear(50, feat_dim),
            nn.Tanh(),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 8 * 4 * 4),
            nn.Tanh(),
            nn.Unflatten(1, (8, 4, 4)),
            nn.ConvTranspose2d(8, 12, 3, 2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(12, 16, 3, 2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 32, 3, 2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 64, 3, 2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 3, 3, 2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CAE(nn.Module):
    #:: CAE
    """CAE"""

    def __init__(self, feat_dim=10):
        super(CAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 6, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 6, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 6, 2, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 1000),
            nn.ReLU(True),
            nn.Linear(1000, feat_dim),
            nn.ReLU(True),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 128 * 14 * 14),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 14, 14)),
            nn.ConvTranspose2d(128, 64, 6, 2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 6, 2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 6, 2, padding=0),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
