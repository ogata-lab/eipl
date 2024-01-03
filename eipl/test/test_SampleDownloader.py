#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.utils import print_info
from eipl.data import SampleDownloader

# get airec dataset
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="HWC")
images, joints = grasp_data.load_raw_data("train")
print_info("Load raw data")
print(
    "images: shape={}, min={:.2f}, max={:.2f}".format(
        images.shape, images.min(), images.max()
    )
)
print(
    "joints: shape={}, min={:.2f}, max={:.2f}".format(
        joints.shape, joints.min(), joints.max()
    )
)

# plot animation
idx = 0
T = images.shape[1]
fig, ax = plt.subplots(1, 2, figsize=(9, 5), dpi=60)


def anim_update(i):
    for j in range(2):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(images[idx, i, :, :, ::-1])
    ax[0].axis("off")
    ax[0].set_title("Input image")

    # plot joint angle
    ax[1].set_ylim(-1.0, 2.0)
    ax[1].set_xlim(0, T)
    ax[1].plot(joints[idx, 1:], linestyle="dashed", c="k")
    for joint_idx in range(8):
        ax[1].plot(np.arange(i + 1), joints[idx, : i + 1, joint_idx])
    ax[1].set_xlabel("Step")
    ax[1].set_title("Joint angles")


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save("./output/viz_downloader.gif")
