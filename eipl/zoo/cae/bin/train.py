#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
import torch
import argparse
from tqdm import tqdm
from collections import OrderedDict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from eipl.model import BasicCAE, CAE, BasicCAEBN, CAEBN
from eipl.data import ImageDataset, SampleDownloader
from eipl.utils import EarlyStopping, check_args, set_logdir

try:
    from libs.trainer import Trainer
except:
    sys.path.append("./libs/")
    from trainer import Trainer


# GPU optimizes and accelerates the network calculations.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# argument parser
parser = argparse.ArgumentParser(description="Learning convolutional autoencoder")
parser.add_argument("--model", type=str, default="CAEBN")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--feat_dim", type=int, default=10)
parser.add_argument("--stdev", type=float, default=0.02)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.1)
parser.add_argument("--vmax", type=float, default=0.9)
parser.add_argument("--device", type=int, default=0)
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
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="CHW")
images, _ = grasp_data.load_norm_data("train", vmin=args.vmin, vmax=args.vmax)
train_dataset = ImageDataset(images, stdev=stdev, training=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)

images, _ = grasp_data.load_norm_data("test", vmin=args.vmin, vmax=args.vmax)
test_dataset = ImageDataset(images, stdev=0.0, training=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)


# define model
if args.model == "BasicCAE":
    model = BasicCAE(feat_dim=args.feat_dim)
elif args.model == "CAE":
    model = CAE(feat_dim=args.feat_dim)
elif args.model == "BasicCAEBN":
    model = BasicCAEBN(feat_dim=args.feat_dim)
elif args.model == "CAEBN":
    model = CAEBN(feat_dim=args.feat_dim)
else:
    assert False, "Unknown model name {}".format(args.model)

# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam.".format(args.optimizer)

# load trainer/tester class
trainer = Trainer(model, optimizer, device=device)

### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "{}.pth".format(args.model))
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch(train_loader)
        test_loss = trainer.process_epoch(test_loader, training=False)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)

        # early stop
        save_ckpt, _ = early_stop(test_loss)

        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name)

        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss, test_loss=test_loss))
