#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import time
import rospy
import numpy as np
from enum import Enum
from sensor_msgs.msg import JointState


def linear_interpolation(q0, q1, T, t):
    """Linear interpolation between q0 and q1"""
    s = min(1.0, max(0.0, t / T))
    return (q1 - q0) * s + q0


class Interpolator:
    def __init__(
        self, control_freq=50, target_freq=10, max_target_jump=20.0 / 180.0 * np.pi
    ):
        self.control_period = 1.0 / control_freq
        self.target_period = 1.0 / target_freq
        self.max_target_jump = max_target_jump
        self.max_target_delay = 2 * self.target_period

        # setup the message
        motor_list = [11, 12, 13, 14, 15]
        self.joint_msg = JointState()
        self.motor_names = ["motor_{}".format(m) for m in motor_list]
        self.joint_msg.name = self.motor_names

        # setup states
        self.has_valid_target = False
        self.target_update_time = None
        self.cmd = None
        self.prev_target = None

        # low freq target input from network
        self.target_sub = rospy.Subscriber(
            "/leader/joint_states", JointState, self.targetCallback
        )

        # setup the online effort controller client
        self.cmd_pub = rospy.Publisher(
            "/leader/interpolated_command", JointState, queue_size=1
        )

    def run(self):
        rate = rospy.Rate(1.0 / self.control_period)

        self.target_update_time = None
        self.target = None
        self.has_valid_target = False

        rospy.logwarn("OnlineExecutor.run(): Starting execution")
        while not rospy.is_shutdown():
            if not self.has_valid_target:
                # just keep the current state
                rospy.logwarn_throttle(
                    1.0, "OnlineExecutor.run(): Wait for first target"
                )

            if self.has_valid_target:
                # interpolation between prev_target and next target
                t = (rospy.Time.now() - self.target_update_time).to_sec()

                self.cmd = linear_interpolation(
                    self.prev_target, self.target, self.target_period, t
                )

                # check if communicaton is broken
                if t > self.max_target_delay:
                    self.has_valid_target = False
                    rospy.logwarn(
                        "OnlineExecutor.run(): Interpolation stopped, wait for valid command"
                    )

                # send the target to robot
                self.command(self.cmd)

            rate.sleep()

        rospy.logwarn("OnlineExecutor.run(): Finished execution")

    def command(self, cmd):
        if not np.isnan(cmd).any():
            self.joint_msg.header.stamp = rospy.Time.now()
            self.joint_msg.position = cmd
            self.cmd_pub.publish(self.joint_msg)

    def targetCallback(self, msg):
        """
        next joint target callback from DL model
        stores next target, current time, previous target
        """

        # extract the state form message
        target = np.array(msg.position)

        if self.cmd is not None:
            # savety first, check last command against new target pose
            if np.max(np.abs(self.cmd - target)) > self.max_target_jump:
                self.has_valid_target = False
                idx = np.argmax(np.abs(self.cmd - target))
                rospy.logerr_throttle(
                    1.0,
                    "OnlineExecutor.targetCallback(): Jump in cmd[{:d}]={:f} to target[{:d}]={:f} > {:f}".format(
                        idx, self.cmd[idx], idx, target[idx], self.max_target_jump
                    ),
                )
                return
        else:
            # initialization
            rospy.logwarn("OnlineExecutor.run(): Recieved first data")
            self.cmd = target

        # store target and last target
        self.target = target
        self.prev_target = np.copy(self.cmd)
        self.target_update_time = rospy.Time.now()

        # target was good
        self.has_valid_target = True


def main(control_freq, target_freq):
    executor = Interpolator(
        control_freq=control_freq,
        target_freq=target_freq,
        max_target_jump=50.0 / 180.0 * np.pi,
    )

    executor.run()


if __name__ == "__main__":
    try:
        rospy.init_node("interpolator_node", anonymous=True)
        control_freq = rospy.get_param("interpolator_node/control_freq")
        target_freq = rospy.get_param("interpolator_node/target_freq")
        main(control_freq, target_freq)
    except rospy.ROSInterruptException:
        pass
