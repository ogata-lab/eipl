#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import time
import torch
import numpy as np
from eipl.data import MultimodalDataset, MultiEpochsDataLoader


def test_dataloader(dataloader):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        for _ in dataloader:
            pass
    torch.cuda.synchronize()
    elapsed_time = time.time() - start
    print(elapsed_time, "sec.")


# parameter
stdev = 0.1
device = "cuda:0"
batch_size = 8
images = np.random.random((30, 200, 3, 64, 64))
joints = np.random.random((30, 200, 7))

# Original dataloader with CUDA transformation
# Note that CUDA transformation does not support pin_memory or num_workers.
dataset = MultimodalDataset(images, joints, device="cuda:0", stdev=stdev)
cuda_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=False
)
test_dataloader(cuda_loader)
del cuda_loader

# Original dataloader with CUDA
dataset = MultimodalDataset(images, joints, device="cpu", stdev=stdev)
cpu_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    prefetch_factor=4,
    num_workers=8,
)
test_dataloader(cpu_loader)
del cpu_loader

# Multiprocess dataloader
dataset = MultimodalDataset(images, joints, device="cpu", stdev=stdev)
multiepoch_loader = MultiEpochsDataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    prefetch_factor=8,
    num_workers=20,
)
test_dataloader(multiepoch_loader)
del multiepoch_loader
