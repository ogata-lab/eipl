#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, default="./data/test/state.npz")
args = parser.parse_args()

# load  sub directory name from filename
savename = "./output/{}_".format(args.filename.split("/")[-2])

# action_infos: action
# obs_infos: image, joint_cos, joint_sin, gripper
data = np.load(args.filename, allow_pickle=True)
if sum(data["success"]) == 0:
    print("[ERROR] This data set has failed task during playback.")
    exit()

# plot joint and images
images = data["images"]
poses = data["poses"]
joints = np.concatenate((data["joints"], data["gripper"]), axis=-1)

N = len(joints)
fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=60)


def anim_update(i):
    for j in range(3):
        ax[j].axis("off")
        ax[j].cla()

    ax[0].imshow(np.flipud(images[i]))
    ax[0].axis("off")
    ax[0].set_title("Image", fontsize=20)

    ax[1].set_ylim(-3.5, 3.5)
    ax[1].set_xlim(0, N)
    for idx in range(8):
        ax[1].plot(np.arange(i + 1), joints[: i + 1, idx])
    ax[1].set_xlabel("Step", fontsize=20)
    ax[1].set_title("Joint angles", fontsize=20)
    ax[1].tick_params(axis="x", labelsize=16)
    ax[1].tick_params(axis="y", labelsize=16)

    ax[2].set_ylim(-3.5, 3.5)
    ax[2].set_xlim(0, N)
    for idx in range(6):
        ax[2].plot(np.arange(i + 1), poses[: i + 1, idx])
    ax[2].set_xlabel("Step", fontsize=20)
    ax[2].set_title("End effector poses", fontsize=20)
    ax[2].tick_params(axis="x", labelsize=16)
    ax[2].tick_params(axis="y", labelsize=16)
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.12, top=0.9)


ani = anim.FuncAnimation(fig, anim_update, interval=int(N / 10), frames=N)
ani.save(savename + "image_joint_ani.gif")

# If an error occurs in generating the gif animation, change the writer (imagemagick/ffmpeg).
# ani.save(savename + "image_joint_ani.gif", writer="imagemagick")
# ani.save(savename + "image_joint_ani.mp4", writer="ffmpeg")
