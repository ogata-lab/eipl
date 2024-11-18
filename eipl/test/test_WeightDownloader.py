#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import shutil
from eipl.data import WeightDownloader

# remove old data
root_dir = os.path.join(os.path.expanduser("~"), ".eipl/")
shutil.rmtree(root_dir)

WeightDownloader("airec", "grasp_bottle")
WeightDownloader("om", "grasp_cube")
