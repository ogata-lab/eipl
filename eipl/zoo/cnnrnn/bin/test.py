#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.model import CNNRNN, CNNRNNLN
from eipl.data import SampleDownloader, WeightDownloader
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
parser.add_argument("--idx", type=str, default="0")
parser.add_argument("--input_param", type=float, default=1.0)
parser.add_argument("--pretrained", action="store_true")
args = parser.parse_args()


# check args
assert args.filename or args.pretrained, "Please set filename or pretrained"

# load pretrained weight
if args.pretrained:
    WeightDownloader("airec", "grasp_bottle")
    args.filename = os.path.join(
        os.path.expanduser("~"), ".eipl/airec/pretrained/CNNRNN/model.pth"
    )

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = int(args.idx)

# load dataset
minmax = [ params['vmin'], params['vmax'] ]
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="HWC")
_images, _joints = grasp_data.load_raw_data("test")
images = _images[idx]
joints = _joints[idx]
joint_bounds = np.load(
    os.path.join(os.path.expanduser("~"), ".eipl/airec/grasp_bottle/joint_bounds.npy")
)
print("images shape:{}, min={}, max={}".format(images.shape, images.min(), images.max()))
print("joints shape:{}, min={}, max={}".format(joints.shape, joints.min(), joints.max()))

# define model
if params["model"] == "CNNRNN":
    model = CNNRNN(rec_dim=params["rec_dim"], joint_dim=8, feat_dim=params["feat_dim"])
elif params["model"] == "CNNRNNLN":
    model = CNNRNNLN(rec_dim=params["rec_dim"], joint_dim=8, feat_dim=params["feat_dim"])
else:
    assert False, "Unknown model name {}".format(params["model"])

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
# Set the inference frequency; for a 10-Hz in ROS system, set as follows.
# freq = 10
# rate = rospy.Rate(freq)
image_list, joint_list = [], []
state = None
nloop = len(images)
for loop_ct in range(nloop):
    # load data and normalization
    img_t = images[loop_ct].transpose(2, 0, 1)
    img_t = torch.Tensor(np.expand_dims(img_t, 0))
    img_t = normalization(img_t, (0, 255), minmax)
    joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))
    joint_t = normalization(joint_t, joint_bounds, minmax)

    # closed loop
    if loop_ct > 0:
        img_t = args.input_param * img_t + (1.0 - args.input_param) * y_image
        joint_t = args.input_param * joint_t + (1.0 - args.input_param) * y_joint

    # predict rnn
    y_image, y_joint, state = model(img_t, joint_t, state)

    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, params["vmin"], params["vmax"])
    pred_image = pred_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, joint_bounds)

    # send pred_joint to robot
    # send_command(pred_joint)
    # pub.publish(pred_joint)

    # Sleep to infer at set frequency.
    # ROS system
    # rate.sleep()
    #
    # Other system
    # time_diff = time.time() - start_time
    # if time_diff < 1./freq:
    #     time.sleep(1./freq - time_diff)

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_image = np.array(image_list)
pred_joint = np.array(joint_list)

# plot animation
T = len(images)
fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)

def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(images[i, :, :, ::-1])
    ax[0].axis("off")
    ax[0].set_title("Input image")

    # plot predicted image
    ax[1].imshow(pred_image[i, :, :, ::-1])
    ax[1].axis("off")
    ax[1].set_title("Predicted image")

    # plot joint angle
    ax[2].set_ylim(-1.0, 2.0)
    ax[2].set_xlim(0, T)
    ax[2].plot(joints[1:], linestyle="dashed", c="k")
    for joint_idx in range(8):
        ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Joint angles")


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save("./output/{}_{}_{}_{}.gif".format(params["model"], params["tag"], idx, args.input_param))

# If an error occurs in generating the gif animation, change the writer (imagemagick/ffmpeg).
#ani.save("./output/{}_{}_{}_{}.gif".format(params["model"], params["tag"], idx, args.input_param), writer="imagemagick")
#ani.save("./output/{}_{}_{}_{}.gif".format(params["model"], params["tag"], idx, args.input_param), writer="ffmpeg")
