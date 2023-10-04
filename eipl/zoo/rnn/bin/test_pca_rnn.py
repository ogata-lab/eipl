#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
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
from sklearn.decomposition import PCA

# own libraries
from eipl.model import BasicLSTM, BasicMTRNN
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))

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
states = []
state = None
N = x_data.shape[1]
for i in range(N):
    _, state = model(x_data[:, i], state)
    # lstm returns hidden state and cell state.
    # Here we store the hidden state to analyze the internal representation of the RNN.
    states.append(state[0])

states = torch.permute(torch.stack(states), (1, 0, 2))
states = tensor2numpy(states)
# Reshape the state from [N,T,D] to [-1,D] for PCA of RNN.
# N is the number of datasets
# T is the sequence length
# D is the dimension of the hidden state
N, T, D = states.shape
states = states.reshape(-1, D)

# plot pca
loop_ct = float(360) / T
pca_dim = 3
pca = PCA(n_components=pca_dim).fit(states)
pca_val = pca.transform(states)
# Reshape the states from [-1, pca_dim] to [N,T,pca_dim] to
# visualize each state as a 3D scatter.
pca_val = pca_val.reshape(N, T, pca_dim)

fig = plt.figure(dpi=60)
ax = fig.add_subplot(projection="3d")


def anim_update(i):
    ax.cla()
    angle = int(loop_ct * i)
    ax.view_init(30, angle)

    c_list = ["C0", "C1", "C2", "C3", "C4"]
    for n, color in enumerate(c_list):
        ax.scatter(
            pca_val[n, 1:, 0], pca_val[n, 1:, 1], pca_val[n, 1:, 2], color=color, s=3.0
        )

    ax.scatter(pca_val[n, 0, 0], pca_val[n, 0, 1], pca_val[n, 0, 2], color="k", s=30.0)
    pca_ratio = pca.explained_variance_ratio_ * 100
    ax.set_xlabel("PC1 ({:.1f}%)".format(pca_ratio[0]))
    ax.set_ylabel("PC2 ({:.1f}%)".format(pca_ratio[1]))
    ax.set_zlabel("PC3 ({:.1f}%)".format(pca_ratio[2]))


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save("./output/PCA_{}_{}.gif".format(params["model"], params["tag"]))

# If an error occurs in generating the gif animation or mp4, change the writer (imagemagick/ffmpeg).
# ani.save("./output/PCA_{}_{}.gif".format(params["model"], params["tag"]), writer="imagemagick")
# ani.save("./output/PCA_{}_{}.mp4".format(params["model"], params["tag"]), writer="ffmpeg")
