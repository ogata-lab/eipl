#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import cv2
import glob
import datetime
import numpy as np
import matplotlib.pylab as plt


def check_path(path, mkdir=False):
    """
    checks given path is existing or not
    """
    if path[-1] == "/":
        path = path[:-1]

    if not os.path.exists(path):
        if mkdir:
            os.mkdir(path)
        else:
            raise ValueError("%s does not exist" % path)
    return path


def set_logdir(log_dir, tag):
    return check_path(os.path.join(log_dir, tag), mkdir=True)
