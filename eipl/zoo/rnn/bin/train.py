#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from eipl.model import BasicLSTM, BasicMTRNN
from eipl.utils import normalization
from eipl.utils import EarlyStopping, check_args, set_logdir

# load own library
sys.path.append("./libs/")
from fullBPTT import fullBPTTtrainer
from dataloader import TimeSeriesDataSet


# argument parser
parser = argparse.ArgumentParser(description="Learning convolutional autoencoder")
parser.add_argument("--model", type=str, default="LSTM")
parser.add_argument("--epoch", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--rec_dim", type=int, default=50)
parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)

# calculate the noise level (variance) from the normalized range
stdev = args.stdev * (args.vmax - args.vmin)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# load dataset
minmax = [args.vmin, args.vmax]
feat_bounds = np.load("../cae/data/feat_bounds.npy")
_feats = np.load("../cae/data/train/features.npy")
train_feats = normalization(_feats, feat_bounds, minmax)
train_joints = np.load("../cae/data/train/joints.npy")
in_dim = train_feats.shape[-1] + train_joints.shape[-1]
train_dataset = TimeSeriesDataSet(
    train_feats, train_joints, minmax=[args.vmin, args.vmax], stdev=stdev
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
)

_feats = np.load("../cae/data/test/features.npy")
test_feats = normalization(_feats, feat_bounds, minmax)
test_joints = np.load("../cae/data/test/joints.npy")
test_dataset = TimeSeriesDataSet(
    test_feats, test_joints, minmax=[args.vmin, args.vmax], stdev=None
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
)

# define model
if args.model == "LSTM":
    model = BasicLSTM(in_dim=in_dim, rec_dim=args.rec_dim, out_dim=in_dim)
elif args.model == "MTRNN":
    model = BasicMTRNN(in_dim, fast_dim=60, slow_dim=5, fast_tau=2, slow_tau=12)
else:
    assert False, "Unknown model name {}".format(args.model)

# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam.".format(
        args.optimizer
    )

# load trainer/tester class
trainer = fullBPTTtrainer(model, optimizer, device=device)

### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "{}.pth".format(args.model))
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch(train_loader)
        with torch.no_grad():
            test_loss = trainer.process_epoch(test_loader, training=False)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)

        # early stop
        save_ckpt, _ = early_stop(test_loss)

        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name)

        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss, test_loss=test_loss))
