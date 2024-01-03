#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn


class CNNRNNLN(nn.Module):
    #:: CNNRNNLN
    """CNNRNNLN"""

    def __init__(self, rec_dim=50, joint_dim=8, feat_dim=10):
        super(CNNRNNLN, self).__init__()

        # Encoder
        self.encoder_image = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, 64, 64]),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 2, 1),
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, 2, 1),
            nn.LayerNorm([16, 16, 16]),
            nn.ReLU(True),
            nn.Conv2d(16, 12, 3, 2, 1),
            nn.LayerNorm([12, 8, 8]),
            nn.ReLU(True),
            nn.Conv2d(12, 8, 3, 2, 1),
            nn.LayerNorm([8, 4, 4]),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 50),
            nn.LayerNorm([50]),
            nn.ReLU(True),
            nn.Linear(50, feat_dim),
            nn.LayerNorm([feat_dim]),
            nn.ReLU(True),
        )

        # Recurrent
        rec_in = feat_dim + joint_dim
        self.rec = nn.LSTMCell(rec_in, rec_dim)

        # Decoder for joint angle
        self.decoder_joint = nn.Sequential(nn.Linear(rec_dim, joint_dim), nn.ReLU(True))

        # Decoder for image
        self.decoder_image = nn.Sequential(
            nn.Linear(rec_dim, 8 * 4 * 4),
            nn.LayerNorm([8 * 4 * 4]),
            nn.ReLU(True),
            nn.Unflatten(1, (8, 4, 4)),
            nn.ConvTranspose2d(8, 12, 3, 2, padding=1, output_padding=1),
            nn.LayerNorm([12, 8, 8]),
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 16, 3, 2, 1, 1),
            nn.LayerNorm([16, 16, 16]),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 3, 2, 1, 1),
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 3, 2, 1, 1),
            nn.LayerNorm([64, 64, 64]),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),
            nn.ReLU(True),
        )

    # image, joint
    def forward(self, xi, xv, state=None):
        # Encoder
        im_feat = self.encoder_image(xi)
        hid = torch.concat([im_feat, xv], -1)

        # Recurrent
        rnn_hid = self.rec(hid, state)

        # Decoder
        y_joint = self.decoder_joint(rnn_hid[0])
        y_image = self.decoder_image(rnn_hid[0])

        return y_image, y_joint, rnn_hid
