#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.utils import normalization


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0)
args = parser.parse_args()

idx = int(args.idx)
joints = np.load("./data/test/joints.npy")
joint_bounds = np.load("./data/joint_bounds.npy")
images = np.load("./data/test/images.npy")
N = images.shape[1]


# normalized joints
minmax = [0.1, 0.9]
norm_joints = normalization(joints, joint_bounds, minmax)

# print data information
print("load test data, index number is {}".format(idx))
print(
    "Joint: shape={}, min={:.3g}, max={:.3g}".format(
        joints.shape, joints.min(), joints.max()
    )
)
print(
    "Norm joint: shape={}, min={:.3g}, max={:.3g}".format(
        norm_joints.shape, norm_joints.min(), norm_joints.max()
    )
)

# plot images and normalized joints
fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=60)


def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot image
    ax[0].imshow(images[idx, i])
    ax[0].axis("off")
    ax[0].set_title("Image", fontsize=20)

    # plot joint angle
    ax[1].set_ylim(-1.0, 2.0)
    ax[1].set_xlim(0, N)
    ax[1].plot(joints[idx], linestyle="dashed", c="k")

    for joint_idx in range(8):
        ax[1].plot(np.arange(i + 1), joints[idx, : i + 1, joint_idx])
    ax[1].set_xlabel("Step", fontsize=20)
    ax[1].set_title("Joint angles", fontsize=20)
    ax[1].tick_params(axis="x", labelsize=16)
    ax[1].tick_params(axis="y", labelsize=16)

    # plot normalized joint angle
    ax[2].set_ylim(0.0, 1.0)
    ax[2].set_xlim(0, N)
    ax[2].plot(norm_joints[idx], linestyle="dashed", c="k")

    for joint_idx in range(8):
        ax[2].plot(np.arange(i + 1), norm_joints[idx, : i + 1, joint_idx])
    ax[2].set_xlabel("Step", fontsize=20)
    ax[2].set_title("Normalized joint angles", fontsize=20)
    ax[2].tick_params(axis="x", labelsize=16)
    ax[2].tick_params(axis="y", labelsize=16)
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.12, top=0.9)


ani = anim.FuncAnimation(fig, anim_update, interval=int(N / 10), frames=N)
ani.save("./output/check_dataset_{}.gif".format(idx))

# If an error occurs in generating the gif animation, change the writer (imagemagick/ffmpeg).
# ani.save("./output/check_dataset_{}.gif".format(idx), writer="imagemagick")
# ani.save("./output/check_dataset_{}.gif".format(idx), writer="ffmpeg")
