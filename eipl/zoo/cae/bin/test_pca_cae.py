#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import torch
import argparse
import matplotlib.pylab as plt
import matplotlib.animation as anim
from sklearn.decomposition import PCA
from eipl.data import SampleDownloader
from eipl.utils import tensor2numpy, restore_args
from eipl.model import BasicCAE, CAE, BasicCAEBN, CAEBN


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))

# load dataset
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="CHW")
images, _ = grasp_data.load_norm_data("test", params["vmin"], params["vmax"])
N, T, C, W, H = images.shape
images = images.reshape(N * T, C, W, H)
images = torch.tensor(images)

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
feat = model.encoder(images)
feat = tensor2numpy(feat)
loop_ct = float(360) / T

pca_dim = 3
pca = PCA(n_components=pca_dim).fit(feat)
pca_val = pca.transform(feat)
pca_val = pca_val.reshape(N, T, pca_dim)

fig = plt.figure()
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


ani = anim.FuncAnimation(fig, anim_update, interval=T, frames=T)
ani.save("./output/PCA_{}_{}.gif".format(params["model"], params["tag"]))

# If an error occurs in generating the gif animation or mp4, change the writer (imagemagick/ffmpeg).
# ani.save("./output/PCA_{}_{}.gif".format(params["model"], params["tag"]), writer="imagemagick")
# ani.save("./output/PCA_{}_{}.mp4".format(params["model"], params["tag"]), writer="ffmpeg")
