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
from eipl.model import BasicCAE, CAE, BasicCAEBN, CAEBN
from eipl.data import SampleDownloader
from eipl.utils import print_info, restore_args, tensor2numpy


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))

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

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

for data_type in ["train", "test"]:
    # generate rnn dataset
    os.makedirs("./data/{}/".format(data_type), exist_ok=True)

    # load dataset
    grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="CHW")
    images, joints = grasp_data.load_norm_data(
        data_type, params["vmin"], params["vmax"]
    )
    images = torch.tensor(images)
    joint_bounds = np.load(
        os.path.join(
            os.path.expanduser("~"), ".eipl/airec/grasp_bottle/joint_bounds.npy"
        )
    )

    # extract image feature
    N = images.shape[0]
    feature_list = []
    for i in range(N):
        _features = model.encoder(images[i])
        feature_list.append(tensor2numpy(_features))

    features = np.array(feature_list)
    np.save("./data/joint_bounds.npy", joint_bounds)
    np.save("./data/{}/features.npy".format(data_type), features)
    np.save("./data/{}/joints.npy".format(data_type), joints)

    print_info("{} data".format(data_type))
    print("==================================================")
    print("Shape of joints angle:", joints.shape)
    print("Shape of image feature:", features.shape)
    print("==================================================")
    print()

# save features minmax bounds
feat_list = []
for data_type in ["train", "test"]:
    feat_list.append(np.load("./data/{}/features.npy".format(data_type)))

feat = np.vstack(feat_list)
feat_minmax = np.array([feat.min(), feat.max()])
np.save("./data/feat_bounds.npy", feat_minmax)
