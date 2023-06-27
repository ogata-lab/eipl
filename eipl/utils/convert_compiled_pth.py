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

restored_ckpt = {}
for k, v in ckpt["model_state_dict"].items():
    restored_ckpt[k.replace("_orig_mod.", "")] = v

ckpt["model_state_dict"] = restored_ckpt

savename = "{}_v1.pth".format(os.path.splitext(args.filename)[0])
torch.save(ckpt, savename)
