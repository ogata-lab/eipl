#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
import argparse
import numpy as np

import robosuite as suite
from robosuite import load_controller_config

sys.path.append("./libs")
from environment import cube
from devices import Keyboard
from utils import input2action
from samplers import BiasedRandomSampler
from rt_control_wrapper import RTControlWrapper


def main(env, nloop=600, save_dir="./data/test"):
    # Set user parameter
    loop_ct = 0

    # Reset the environment and Setup rendering
    obs = env.reset()
    initial_q = np.deg2rad([0.0, 12.0, 0.0, -150.0, 0.0, 168.0, 45])
    env.set_joint_qpos(initial_q)
    env.sim.forward()
    env.render()

    # Initialize device control
    device.start_control()

    # Set active robot
    active_robot = env.robots["right" == "left"]

    # save list
    rewards = []
    timestamp = []
    states = []
    joints = []
    action_infos = []  # stores information about actions taken
    obs_infos = []  # stores information about observation taken
    successful = []  # stores success state of demonstration

    while True:
        # Get the newest action
        action, _, recording_state = input2action(
            device=device,
            robot=active_robot,
            active_arm="right",
            env_configuration=None,
        )

        # get robot state
        obs, reward, _, _ = env.step(action)
        obs.pop("agentview_image")

        state, success = env.get_state()
        joint = env.get_joints()
        env.render()

        if recording_state:
            print("loop_ct [step]: {}/{}".format(loop_ct, nloop))

            # rewards
            rewards.append(reward)

            # save states and successful
            joints.append(joint)
            states.append(state)
            successful.append(success)

            # save action
            action_info = {}
            action_info["actions"] = np.array(action)
            action_infos.append(action_info)

            # save observation
            obs_infos.append(obs)

            loop_ct += 1
            if loop_ct >= nloop:
                break

        if recording_state == -1:
            break

    # saveing
    print("save fille: {}/state.npz".format(save_dir))
    np.savez(
        os.path.join(save_dir, "state.npz"),
        nloop=nloop,
        rewards=np.array(rewards),
        timestamp=np.array(timestamp),
        states=np.array(states),
        action_infos=action_infos,
        obs_infos=obs_infos,
        joint_angles=np.array(joints),
        successful=np.array(successful),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=float, default=0.0)
    parser.add_argument("--nloop", type=int, default=600)
    parser.add_argument("--ep_dir", type=str, default="./data/raw_data/test")
    args = parser.parse_args()

    # Get controller config
    # Import controller config IK_POSE/OSC_POSE/OSC_POSITION/JOINT_POSITION
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # Create argument configurationstate
    config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,
    }

    # train position: collect 7 data/position (5 for train, 2 for test)
    # test  position: collect 2 data/position (all for test)
    #            Pos1 Pos2  Pos3  Pos4 Pos5 Pos6 Pos7 Pos8 Pos9
    # pos_train: -0.2       -0.1       0.0       0.1       0.2
    # pos_test:  -0.2 -0.15 -0.1 -0.05 0.0  0.05 0.1  0.15 0.2

    position_sampler = BiasedRandomSampler(
        name="ObjectSampler",
        mujoco_objects=cube,
        x_range=[-0.03, 0.03],
        # y_range=[-0.03, 0.03],
        rotation=False,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01,
        pos_bias_list=[[0.0, args.pos]],
    )

    # create original environment
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
    env = RTControlWrapper(env, args.ep_dir, vis_settings=True)

    # How much to scale position/rotation user inputs
    device = Keyboard(pos_sensitivity=0.05, rot_sensitivity=0.05)
    env.viewer.add_keypress_callback(device.on_press)

    # collect some data
    print("Collecting demonstration data ...")
    main(env, nloop=args.nloop, save_dir=args.ep_dir)
