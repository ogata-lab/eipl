#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, default=None)
args = parser.parse_args()

# resave original file
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
savename = "{}_org.pth".format(os.path.splitext(args.filename)[0])
torch.save(ckpt, savename)

# save pth file without optimizer state
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
ckpt.pop("optimizer_state_dict")
torch.save(ckpt, args.filename)
