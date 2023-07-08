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
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from eipl.model import SARNN
from eipl.data import MultimodalDataset, SampleDownloader
from eipl.utils import EarlyStopping, check_args, set_logdir

try:
    from libs.fullBPTT import fullBPTTtrainer
except:
    sys.path.append("./libs/")
    from fullBPTT import fullBPTTtrainer


# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model", type=str, default="sarnn")
parser.add_argument("--epoch", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--rec_dim", type=int, default=50)
parser.add_argument("--k_dim", type=int, default=5)
parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=0.1)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)
parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
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
images, joints = grasp_data.load_norm_data("train", vmin=args.vmin, vmax=args.vmax)
train_dataset = MultimodalDataset(images, joints, stdev=stdev, training=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)

images, joints = grasp_data.load_norm_data("test", vmin=args.vmin, vmax=args.vmax)
test_dataset = MultimodalDataset(images, joints, stdev=0.0, training=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)


# define model
model = SARNN(
    rec_dim=args.rec_dim,
    joint_dim=8,
    k_dim=args.k_dim,
    heatmap_size=args.heatmap_size,
    temperature=args.temperature,
)

# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam.".format(args.optimizer)

# load trainer/tester class
loss_weights = [args.img_loss, args.joint_loss, args.pt_loss]
trainer = fullBPTTtrainer(model, optimizer, loss_weights=loss_weights, device=device)

### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "SARNN.pth")
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
