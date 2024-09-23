#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import cv2
import time
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.utils import deprocess_img
from eipl.data import SampleDownloader, ImageDataset, MultimodalDataset


# Download dataset
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="CHW")
images, joints = grasp_data.load_norm_data("train", vmin=0.1, vmax=0.9)

# test for ImageDataset
dataset = ImageDataset(images, stdev=0.02)
x_img, y_img = dataset[0]
x_im = x_img.numpy()[::-1]
y_im = y_img.numpy()[::-1]

plt.figure(dpi=60)
plt.subplot(1, 2, 1)
plt.imshow(x_im.transpose(1, 2, 0))
plt.axis("off")
plt.title("Input image")
plt.subplot(1, 2, 2)
plt.imshow(y_im.transpose(1, 2, 0))
plt.title("True image")
plt.axis("off")
plt.savefig("./output/viz_image_dataset.png")
plt.close()

# test for MultimodalDataset
multi_dataset = MultimodalDataset(images, joints)
x_data, y_data = multi_dataset[1]
x_img = x_data[0]
y_img = y_data[0]

# tensor to numpy
x_img = deprocess_img(x_img.numpy().transpose(0, 2, 3, 1), 0.1, 0.9)
y_img = deprocess_img(y_img.numpy().transpose(0, 2, 3, 1), 0.1, 0.9)


# plot images
T = len(x_img)
fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=60)


def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot predicted image
    ax[0].imshow(y_img[i, :, :, ::-1])
    ax[0].axis("off")
    ax[0].set_title("Original image", fontsize=20)

    # plot camera image
    ax[1].imshow(x_img[i, :, :, ::-1])
    ax[1].axis("off")
    ax[1].set_title("Noisied image", fontsize=20)

    # plot joint angle
    ax[2].set_ylim(0.0, 1.0)
    ax[2].set_xlim(0, T)
    ax[2].plot(y_data[1], linestyle="dashed", c="k")
    for joint_idx in range(8):
        ax[2].plot(np.arange(i + 1), x_data[1][: i + 1, joint_idx])
    ax[2].set_xlabel("Step", fontsize=20)
    ax[2].set_title("Scaled joint angles", fontsize=20)
    ax[2].tick_params(axis="x", labelsize=16)
    ax[2].tick_params(axis="y", labelsize=16)
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.12, top=0.9)


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save("./output/viz_multimodal_dataset.gif")
