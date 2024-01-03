#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
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
from rt_control_wrapper import RTControlWrapper


def playback(env, ep_dir, rate):
    data = np.load(os.path.join(ep_dir, "state.npz"), allow_pickle=True)
    nloop = data["nloop"]
    states = data["states"]
    joints = data["joint_angles"]
    grippers = np.array([el["actions"][-1] for el in data["action_infos"]])

    # Set user parameter
    loop_ct = 0

    # Reset the environment and Set environment
    env.set_environment(ep_dir, states[0])

    # save list
    pose_list = []
    state_list = []
    joint_list = []
    image_list = []
    gripper_list = []
    success_list = []

    for loop_ct in range(nloop):
        # append
        if loop_ct % rate == 0:
            action = env.get_joint_action(joints[loop_ct], kp=rate)
            action[-1] = grippers[loop_ct]

        obs, reward, done, info = env.step(action)
        # env.render()

        # save data
        if loop_ct % rate == 0:
            state, success = env.get_state()
            image = env.get_image()
            joint = env.get_joints()
            pose = env.get_pose()

            # save robot sensor data
            pose_list.append(pose)
            state_list.append(state)
            image_list.append(image)
            joint_list.append(joint)
            success_list.append(success)
            gripper_list.append(action[-1])

    # saveing
    save_name = os.path.join(ep_dir, "state_resave.npz")
    print("save fille: ", save_name)
    np.savez(
        save_name,
        poses=np.array(pose_list),
        states=np.array(state_list),
        joints=np.array(joint_list),
        images=np.array(image_list),
        success=np.array(success_list),
        gripper=np.array(gripper_list).reshape(-1, 1),
    )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_dir", type=str, default="./data/raw_data/test/")
    parser.add_argument("--rate", type=int, default=5)  # Down sampling rate
    args = parser.parse_args()

    # Get controller config
    # Import controller config IK_POSE/OSC_POSE/OSC_POSITION/JOINT_POSITION
    controller_config = load_controller_config(default_controller="JOINT_POSITION")

    # Create argument configuration
    config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,
    }

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
    )

    # wrap the environment with data collection wrapper
    env = RTControlWrapper(env)

    # collect some data
    playback(env, args.ep_dir, args.rate)
