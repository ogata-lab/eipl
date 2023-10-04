#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import numpy as np
from torch.utils.data import Dataset


class TimeSeriesDataSet(Dataset):
    """AIREC_sample dataset.

    Args:
        feats (np.array):  Set the image features.
        joints (np.array): Set the joint angles.
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
    """

    def __init__(self, feats, joints, minmax=[0.1, 0.9], stdev=0.0):
        self.stdev = stdev
        self.feats = torch.from_numpy(feats).float()
        self.joints = torch.from_numpy(joints).float()

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        # normalization and convert numpy array to torch tensor
        y_feat = self.feats[idx]
        y_joint = self.joints[idx]
        y_data = torch.concat((y_feat, y_joint), axis=-1)

        # apply gaussian noise to joint angles and image features
        if self.stdev is not None:
            x_feat = self.feats[idx] + torch.normal(
                mean=0, std=self.stdev, size=y_feat.shape
            )
            x_joint = self.joints[idx] + torch.normal(
                mean=0, std=self.stdev, size=y_joint.shape
            )
        else:
            x_feat = self.feats[idx]
            x_joint = self.joints[idx]

        x_data = torch.concat((x_feat, x_joint), axis=-1)

        return [x_data, y_data]


if __name__ == "__main__":
    import time

    # random dataset
    feats = np.random.randn(10, 120, 10)
    joints = np.random.randn(10, 120, 8)

    # load data
    data_loader = TimeSeriesDataSet(feats, joints, minmax=[0.1, 0.9])
    x_data, y_data = data_loader[1]
    print(x_data.shape, y_data.shape)

    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=3,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    print("[Start] load data using torch data loader")
    start_time = time.time()
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)

    print("[Finish] time: ", time.time() - start_time)
