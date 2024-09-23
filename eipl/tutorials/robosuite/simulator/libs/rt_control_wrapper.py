#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import numpy as np
import transforms3d as T
from robosuite.wrappers import Wrapper


class RTControlWrapper(Wrapper):
    def __init__(self, env, save_dir=None, vis_settings=False):
        super().__init__(env)
        """
        RTControlWrapper specialized for imitation learning
        """

        # initialization
        self.robot = self.env.robots[0]
        self.robot.controller_config["kp"] = 150

        # save the task instance (will be saved on the first env interaction)
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())

        # make save directory and save xml file
        if save_dir is not None:
            if not os.path.exists(save_dir):
                print(
                    "DataCollectionWrapper: making new directory at {}".format(save_dir)
                )
                os.makedirs(save_dir)
            self.save_xml(save_dir)

        self._vis_settings = None
        if vis_settings:
            # Create internal dict to store visualization settings (set to True by default)
            self._vis_settings = {vis: True for vis in self.env._visualizations}

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection
        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)
        if self._vis_settings is not None:
            self.env.visualize(vis_settings=self._vis_settings)

        return ret

    def save_xml(self, save_dir="./data"):
        # save the model xml
        xml_path = os.path.join(save_dir, "model.xml")
        with open(xml_path, "w") as f:
            f.write(self._current_task_instance_xml)

    def set_environment(self, save_dir, state):
        xml_path = os.path.join(save_dir, "model.xml")
        with open(xml_path, "r") as f:
            self.env.reset_from_xml_string(f.read())

        self.env.sim.set_state_from_flattened(state)
        if self._vis_settings is not None:
            self.env.visualize(vis_settings=self._vis_settings)

    def set_gripper_qpos(self, qpos):
        for i, q in enumerate(qpos):
            self.env.sim.data.set_joint_qpos(
                name="gripper0_finger_joint{}".format(i + 1), value=q
            )

    def set_joint_qpos(self, qpos):
        for i, q in enumerate(qpos):
            self.env.sim.data.set_joint_qpos(
                name="robot0_joint{}".format(i + 1), value=q
            )

    def get_image(self, name="agentview_image"):
        return self.env.observation_spec()[name]

    def get_joint_action(self, goal_joint_pos, kp, kd=0.0):
        """relative2absolute_joint_pos_commands"""
        action = [0 for _ in range(self.robot.dof)]
        curr_joint_pos = self.robot._joint_positions
        curr_joint_vel = self.robot._joint_velocities

        for i in range(len(goal_joint_pos)):
            action[i] = (goal_joint_pos[i] - curr_joint_pos[i]) * kp - curr_joint_vel[
                i
            ] * kd

        return action

    def get_pose(self):
        position = self.env.robots[0]._hand_pos
        orientation_matrix = self.env.robots[0]._hand_orn
        orientation_euler = T.euler.mat2euler(orientation_matrix)

        pose = list(position) + list(orientation_euler)
        pose = np.array(pose)
        pose[pose < -np.pi / 2] += np.pi * 2

        return pose

    def get_gripper(self):
        return self.env.observation_spec()["robot0_gripper_qpos"]

    def get_joints(self):
        return self.robot._joint_positions

    def check_success(self):
        return self.env._check_success()

    def get_state(self):
        state = self.env.sim.get_state().flatten()
        # successful
        if self.env._check_success():
            return state, True
        else:
            return state, False
