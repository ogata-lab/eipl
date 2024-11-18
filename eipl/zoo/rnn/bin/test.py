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

# own libraries
from eipl.model import BasicLSTM, BasicMTRNN
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy
from eipl.data import WeightDownloader


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
        os.path.expanduser("~"), ".cache/eipl/airec/grasp_bottle/weights/RNN/model.pth"
    )

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = int(args.idx)

# load dataset
minmax = [params["vmin"], params["vmax"]]
feat_bounds = np.load("../cae/data/feat_bounds.npy")
_feats = np.load("../cae/data/test/features.npy")
test_feats = normalization(_feats, feat_bounds, minmax)
test_joints = np.load("../cae/data/test/joints.npy")
x_data = np.concatenate((test_feats, test_joints), axis=-1)
x_data = torch.Tensor(x_data)
in_dim = x_data.shape[-1]

# define model
if params["model"] == "LSTM":
    model = BasicLSTM(in_dim=in_dim, rec_dim=params["rec_dim"], out_dim=in_dim)
elif params["model"] == "MTRNN":
    model = BasicMTRNN(in_dim, fast_dim=60, slow_dim=5, fast_tau=2, slow_tau=12)
else:
    assert False, "Unknown model name {}".format(params["model"])

if params["compile"]:
    model = torch.compile(model)

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
y_hat = []
state = None
T = x_data.shape[1]
for i in range(T):
    _y, state = model(x_data[:, i], state)
    y_hat.append(_y)

y_hat = torch.permute(torch.stack(y_hat), (1, 0, 2))
y_hat = tensor2numpy(y_hat)
y_joints = y_hat[:, :, 10:]
y_feats = y_hat[:, :, :10]

# plot animation
fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=60)


def anim_update(i):
    for j in range(2):
        ax[j].cla()

    ax[0].set_ylim(-0.1, 1.1)
    ax[0].set_xlim(0, T)
    ax[0].plot(test_joints[idx, 1:], linestyle="dashed", c="k")
    for joint_idx in range(8):
        ax[0].plot(np.arange(i + 1), y_joints[idx, : i + 1, joint_idx])
    ax[0].set_xlabel("Step")
    ax[0].set_title("Joint angles")

    ax[1].set_ylim(-0.1, 1.1)
    ax[1].set_xlim(0, T)
    ax[1].plot(test_feats[idx, 1:], linestyle="dashed", c="k")
    for joint_idx in range(10):
        ax[1].plot(np.arange(i + 1), y_feats[idx, : i + 1, joint_idx])
    ax[1].set_xlabel("Step")
    ax[1].set_title("Image features")


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save("./output/{}_{}_{}.gif".format(params["model"], params["tag"], idx))

# If an error occurs in generating the gif animation or mp4, change the writer (imagemagick/ffmpeg).
# ani.save("./output/{}_{}_{}.gif".format(params["model"], params["tag"], idx), writer="imagemagick")
# ani.save("./output/{}_{}_{}.mp4".format(params["model"], params["tag"], idx), writer="ffmpeg")
