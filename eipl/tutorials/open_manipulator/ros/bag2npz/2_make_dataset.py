#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pylab as plt
from utils import resize_img, calc_minmax, list_to_numpy


def load_data(dir):
    joints = []
    images = []
    seq_length = []

    files = glob.glob(os.path.join(dir, "*.npz"))
    files.sort()

    for filename in files:
        print(filename)
        npz_data = np.load(filename)
        images.append(resize_img(npz_data["images"], (64, 64)))
        _joints = npz_data["joints"]
        joints.append(_joints)
        seq_length.append(len(_joints))

    max_seq = max(seq_length)
    images = list_to_numpy(images, max_seq)
    joints = list_to_numpy(joints, max_seq)

    return images, joints


if __name__ == "__main__":
    train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    test_list = [9, 10, 11, 12, 13]

    # load data
    images, joints = load_data("./bag/")

    # save images and joints
    np.save("./data/train/images.npy", images[train_list].astype(np.uint8))
    np.save("./data/train/joints.npy", joints[train_list].astype(np.float32))
    np.save("./data/test/images.npy", images[test_list].astype(np.uint8))
    np.save("./data/test/joints.npy", joints[test_list].astype(np.float32))

    # save joint bounds
    joint_bounds = calc_minmax(joints)
    np.save("./data/joint_bounds.npy", joint_bounds)
