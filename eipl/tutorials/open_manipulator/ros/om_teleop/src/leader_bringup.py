#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
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


class LeaderARM(DynamixelDriver):
    def __init__(self, freq, motor_list, device="/dev/ttyUSB1", baudrate=1000000):
        self.freq = freq
        self.motor_list = motor_list
        self.motor_names = ["motor_{}".format(m) for m in motor_list]
        self.baudrate = baudrate
        self.joint_pub = rospy.Publisher(
            "/leader/joint_states", JointState, queue_size=1
        )

        # setup the message
        self.joint_msg = JointState()
        self.joint_msg.name = self.motor_names

        # Initialization
        self.portHandler = PortHandler(device)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.groupBulkWrite = GroupBulkWrite(self.portHandler, self.packetHandler)
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)
        self.serial_open()

    def run(self):
        rate = rospy.Rate(self.freq)

        rospy.logwarn("LeaderARM.run(): Starting execution")
        while not rospy.is_shutdown():
            # set message
            self.joint_msg.header.stamp = rospy.Time.now()
            self.joint_msg.position = self.get_joint_radian()

            # publish
            self.joint_pub.publish(self.joint_msg)
            rate.sleep()

        rospy.logwarn("LeaderARM.run(): Finished execution")
        self.torque_off()


def main(freq, motor_list, device, baudrate):
    leader_arm = LeaderARM(freq, motor_list, device, baudrate)

    leader_arm.run()


if __name__ == "__main__":
    try:
        rospy.init_node("dxl_leader_node", anonymous=True)
        freq = rospy.get_param("dxl_leader_node/freq")
        device = rospy.get_param("dxl_leader_node/device")
        baudrate = rospy.get_param("dxl_leader_node/baudrate")
        motor_list = rospy.get_param("dxl_leader_node/motor_list")
        motor_list = [eval(m) for m in motor_list.split(",")]
        main(freq, motor_list, device, baudrate)
    except rospy.ROSInterruptException:
        pass
