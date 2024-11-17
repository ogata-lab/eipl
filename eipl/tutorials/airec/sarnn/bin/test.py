#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import argparse
import os
from pathlib import Path

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

from eipl.data import SampleDownloader, WeightDownloader
from eipl.model import SARNN
from eipl.utils import (
    deprocess_img,
    normalization,
    resize_img,
    restore_args,
    tensor2numpy,
)

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
parser.add_argument("--idx", type=str, default="0")
parser.add_argument("--input_param", type=float, default=1.0)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--close-joint", action="store_true")
args = parser.parse_args()

# check args
assert args.filename or args.pretrained, "Please set filename or pretrained"

# load pretrained weight
if args.pretrained:
    WeightDownloader("airec", "grasp_bottle")
    args.filename = os.path.join(
        os.path.expanduser("~"),
        ".cache/eipl/airec/grasp_bottle/weights/SARNN/model.pth",
    )

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = int(args.idx)

# load dataset
minmax = [params["vmin"], params["vmax"]]
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="HWC")
_images, _joints = grasp_data.load_raw_data("test")
_images = resize_img(_images, (64, 64))
images = _images[idx]
joints = _joints[idx]
joint_bounds = grasp_data.joint_bounds
print(
    "images shape:{}, min={}, max={}".format(images.shape, images.min(), images.max())
)
print(
    "joints shape:{}, min={}, max={}".format(joints.shape, joints.min(), joints.max())
)

# define model
model = SARNN(
    rec_dim=params["rec_dim"],
    joint_dim=8,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
)

if params["compile"]:
    model = torch.compile(model)

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
img_size = 128
image_list, joint_list = [], []
ect_pts_list, dec_pts_list = [], []
state = None
nloop = len(images)
for loop_ct in range(nloop):
    # load data and normalization
    img_t = images[loop_ct].transpose(2, 0, 1)
    img_t = torch.Tensor(np.expand_dims(img_t, 0))
    img_t = normalization(img_t, (0, 255), minmax)
    joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))
    joint_t = normalization(joint_t, joint_bounds, minmax)

    # closed loop
    if loop_ct > 0:
        img_t = args.input_param * img_t + (1.0 - args.input_param) * y_image
        joint_t = args.input_param * joint_t + (1.0 - args.input_param) * y_joint

    if args.close_joint and loop_ct > 0:
        joint_t = y_joint

    # predict rnn
    y_image, y_joint, ect_pts, dec_pts, state = model(img_t, joint_t, state)

    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, params["vmin"], params["vmax"])
    pred_image = pred_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, joint_bounds)

    # TODO: send pred_joint to robot
    # send_command(pred_joint)
    # pub.publish(pred_joint)

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)
    ect_pts_list.append(tensor2numpy(ect_pts[0]))
    dec_pts_list.append(tensor2numpy(dec_pts[0]))

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_image = np.array(image_list)
pred_joint = np.array(joint_list)

# split key points
ect_pts = np.array(ect_pts_list)
dec_pts = np.array(dec_pts_list)
ect_pts = ect_pts.reshape(-1, params["k_dim"], 2) * img_size
dec_pts = dec_pts.reshape(-1, params["k_dim"], 2) * img_size
ect_pts = np.clip(ect_pts, 0, img_size)
dec_pts = np.clip(dec_pts, 0, img_size)


# plot images
T = len(images)
fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)


def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(images[i, :, :, ::-1])
    for j in range(params["k_dim"]):
        ax[0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1], "bo", markersize=6)  # encoder
        ax[0].plot(
            dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
        )  # decoder
    ax[0].axis("off")
    ax[0].set_title("Input image")

    # plot predicted image
    ax[1].imshow(pred_image[i, :, :, ::-1])
    ax[1].axis("off")
    ax[1].set_title("Predicted image")

    # plot joint angle
    ax[2].set_ylim(-1.0, 2.0)
    ax[2].set_xlim(0, T)
    ax[2].plot(joints[1:], linestyle="dashed", c="k")
    for joint_idx in range(8):
        ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Joint angles")


ani = anim.FuncAnimation(fig, anim_update, frames=T, interval=100)
animation_path = (
    Path("output")
    / params["tag"]
    / f"SARNN_{idx}{'_close' if args.close_joint else ''}.mp4"
)
if animation_path.parent.exists():
    animation_path.parent.mkdir(parents=True)
ani.save(animation_path, writer="ffmpeg", fps=10)
