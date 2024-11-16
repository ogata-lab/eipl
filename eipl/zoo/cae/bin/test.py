#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.model import BasicCAE, CAE, BasicCAEBN, CAEBN
from eipl.data import SampleDownloader, WeightDownloader
from eipl.utils import normalization, deprocess_img, restore_args
from eipl.utils import tensor2numpy


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
parser.add_argument("--idx", type=str, default="0")
parser.add_argument("--pretrained", action="store_true")
args = parser.parse_args()

# check args
assert args.filename or args.pretrained, "Please set filename or pretrained"

# load pretrained weight
if args.pretrained:
    WeightDownloader("airec", "grasp_bottle")
    args.filename = os.path.join(
        os.path.expanduser("~"), ".cache/eipl/airec/grasp_bottle/weights/CAEBN/model.pth"
    )

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = int(args.idx)

# load dataset
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="HWC")
images_raw, _ = grasp_data.load_raw_data("test")
images = normalization(
    images_raw.astype(np.float32), (0.0, 255.0), (params["vmin"], params["vmax"])
)
images = images.transpose(0, 1, 4, 2, 3)
images = torch.tensor(images)
T = images.shape[1]

# define model
if params["model"] == "BasicCAE":
    model = BasicCAE(feat_dim=params["feat_dim"])
elif params["model"] == "CAE":
    model = CAE(feat_dim=params["feat_dim"])
elif params["model"] == "BasicCAEBN":
    model = BasicCAEBN(feat_dim=params["feat_dim"])
elif params["model"] == "CAEBN":
    model = CAEBN(feat_dim=params["feat_dim"])
else:
    assert False, "Unknown model name {}".format(params["model"])

if params["compile"]:
    model = torch.compile(model)

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# prediction
_yi = model(images[idx])
yi = deprocess_img(tensor2numpy(_yi), params["vmin"], params["vmax"])
yi = yi.transpose(0, 2, 3, 1)

# plot images
fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=60)


def anim_update(i):
    for j in range(2):
        ax[j].cla()

    ax[0].imshow(images_raw[idx, i, :, :, ::-1])
    ax[0].axis("off")
    ax[0].set_title("Input image")

    ax[1].imshow(yi[i, :, :, ::-1])
    ax[1].axis("off")
    ax[1].set_title("Reconstructed image")


# defaults
ani = anim.FuncAnimation(fig, anim_update, interval=int(T / 10), frames=T)
ani.save("./output/{}_{}_{}.gif".format(params["model"], params["tag"], idx))

# If an error occurs in generating the gif animation or mp4, change the writer (imagemagick/ffmpeg).
# ani.save("./output/{}_{}_{}.gif".format(params["model"], params["tag"], idx), writer="imagemagick")
# ani.save("./output/{}_{}_{}.mp4".format(params["model"], params["tag"], idx), writer="ffmpeg")
