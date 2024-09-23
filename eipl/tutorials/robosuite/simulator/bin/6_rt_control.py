#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import cv2
import sys
import torch
import argparse
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from eipl.model import SARNN
from eipl.utils import tensor2numpy, normalization, deprocess_img, restore_args

sys.path.append("./libs")
from environment import cube
from samplers import BiasedRandomSampler
from rt_control_wrapper import RTControlWrapper


# load param
joint_bounds = np.load("./data/joint_bounds.npy")


def rt_control(env, nloop, rate):
    # Set user parameter
    loop_ct = 0

    # Reset the environment and Set environment
    obs = env.reset()
    initial_q = np.deg2rad([0.0, 12.0, 0.0, -150.0, 0.0, 168.0, 45])
    env.set_joint_qpos(initial_q)
    env.sim.forward()

    gripper_command = -1.0
    gripper_state = np.array([0.0])
    state = None

    for loop_ct in range(nloop):
        if loop_ct % rate == 0:
            # get image
            img = env.get_image()
            img = cv2.resize(img[::-1], (64, 64))
            img = np.expand_dims(img.transpose(2, 0, 1), 0)
            xi = normalization(img.astype(np.float32), (0, 255), minmax)

            # get joint and gripper angles
            joint = env.get_joints()
            xv = np.concatenate((joint, gripper_state), axis=-1)
            xv = np.expand_dims(xv, 0)
            xv = normalization(xv, joint_bounds, minmax)

            # rt predict
            xi = torch.Tensor(xi)
            xv = torch.Tensor(xv)
            _yi, _yv, enc_ij, dec_ij, state = model(xi, xv, state)

            # post process
            yv = tensor2numpy(_yv[0])
            yv = normalization(yv, minmax, joint_bounds)

            if yv[-1] > 0.7 and gripper_command == -1.0:
                gripper_command = 1.0
            if yv[-1] < 0.3 and gripper_command == 1.0:
                gripper_command = -1.0
            gripper_state = yv[-1:]

            action = env.get_joint_action(yv[:-1], kp=rate)
            action[-1] = gripper_command

        env.step(action)
        env.render()
        _, success = env.get_state()
        if success:
            break

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Model name")
    parser.add_argument("--rate", type=int, default=5)  # Down sampling rate
    args = parser.parse_args()

    dir_name = os.path.split(args.filename)[0]
    params = restore_args(os.path.join(dir_name, "args.json"))
    minmax = [params["vmin"], params["vmax"]]

    # define model
    model = SARNN(
        rec_dim=params["rec_dim"],
        joint_dim=8,
        k_dim=params["k_dim"],
        heatmap_size=params["heatmap_size"],
        temperature=params["temperature"],
        im_size=[64, 64],
    )

    if params["compile"]:
        model = torch.compile(model)

    # load weight
    ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get controller config
    # Import controller config IK_POSE/OSC_POSE/OSC_POSITION/JOINT_POSITION
    controller_config = load_controller_config(default_controller="JOINT_POSITION")

    # Create argument configuration
    config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,
    }

    # train position x7 (5 for train, 2 for test)
    # test  position x2 (all for test)
    #            Pos1 Pos2  Pos3  Pos4 Pos5 Pos6 Pos7 Pos8 Pos9
    # pos_train: -0.2       -0.1       0.0       0.1       0.2
    # pos_test:  -0.2 -0.15 -0.1 -0.05 0.0  0.05 0.1  0.15 0.2
    # reinforcement: -0.3/0.3

    x_pos = normalization(np.random.random(), (0, 1), (-0.2, 0.2))
    position_sampler = BiasedRandomSampler(
        name="ObjectSampler",
        mujoco_objects=cube,
        rotation=False,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01,
        pos_bias_list=[[0.0, x_pos]],
    )

    # create original environment
    # create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
        placement_initializer=position_sampler,
    )

    # wrap the environment with data collection wrapper
    env = RTControlWrapper(env)

    total_loop = 10
    # realtime contol
    for loop_ct in range(total_loop):
        res = rt_control(env, nloop=600, rate=args.rate)
        if res:
            print("[{}/{}] Task succeeded!".format(loop_ct + 1, total_loop))
        else:
            print("[{}/{}] Task failed!".format(loop_ct + 1, total_loop))
