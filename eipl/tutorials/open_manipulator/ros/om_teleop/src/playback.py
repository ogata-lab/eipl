#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import sys
import time
import numpy as np
from dynamixel_sdk import *
from dynamixel_driver import *

# ROS
import rospy
from sensor_msgs.msg import JointState


def main(freq, motor_list):
    # initialization
    motor_list = motor_list
    motor_names = ["motor_{}".format(m) for m in motor_list]

    # publisher
    joint_pub = rospy.Publisher("/leader/joint_states", JointState, queue_size=1)

    # setup the message
    joint_msg = JointState()
    joint_msg.name = motor_names

    # load_data
    joint_angles = np.load("../bag2npy/data/train/joints.npy")[0]
    nloop = len(joint_angles)
    rate = rospy.Rate(freq)

    rospy.logwarn("Playback: Starting execution")
    for loop_ct in range(nloop):
        # set message
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.position = joint_angles[loop_ct]

        # publish
        joint_pub.publish(joint_msg)
        rate.sleep()

    rospy.logwarn("Playback: Finished execution")


if __name__ == "__main__":
    try:
        rospy.init_node("dxl_playback_node", anonymous=True)
        freq = rospy.get_param("dxl_playback_node/freq")
        motor_list = rospy.get_param("dxl_playback_node/motor_list")
        motor_list = [eval(m) for m in motor_list.split(",")]
        main(freq, motor_list)
    except rospy.ROSInterruptException:
        pass
