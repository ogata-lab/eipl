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


class FollowerARM(DynamixelDriver):
    def __init__(self, freq, motor_list, device="/dev/ttyUSB0", baudrate=1000000):
        self.freq = freq
        self.motor_list = motor_list
        self.motor_names = ["motor_{}".format(m) for m in motor_list]
        self.baudrate = baudrate
        self.cmd_sub = rospy.Subscriber(
            "/leader/interpolated_command", JointState, self.cmdCallback
        )
        self.joint_pub = rospy.Publisher(
            "/follower/joint_states", JointState, queue_size=1
        )

        self.cmd = None

        # setup the message
        self.joint_msg = JointState()
        self.joint_msg.name = self.motor_names

        # Initialization
        self.portHandler = PortHandler(device)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.groupBulkWrite = GroupBulkWrite(self.portHandler, self.packetHandler)
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)
        self.serial_open()
        self.set_gripper_mode()
        self.torque_on()

    def cmdCallback(self, msg):
        self.cmd = np.array(msg.position)

    def run(self):
        rate = rospy.Rate(self.freq)

        rospy.logwarn("FollowerARM.run(): Starting execution")
        while not rospy.is_shutdown():
            # start_time = time.time()
            # set joint
            if self.cmd is not None:
                self.set_joint_radian(self.cmd)

            # set message
            self.joint_msg.header.stamp = rospy.Time.now()
            self.joint_msg.position = self.get_joint_radian()

            # publish
            self.joint_pub.publish(self.joint_msg)
            rate.sleep()
            # print(time.time()-start_time)

        rospy.logwarn("FollowerARM.run(): Finished execution")
        self.torque_off()


def main(freq, motor_list, device, baudrate):
    follower_arm = FollowerARM(freq, motor_list, device, baudrate)

    follower_arm.run()


if __name__ == "__main__":
    try:
        rospy.init_node("dxl_follower_node", anonymous=True)
        freq = rospy.get_param("dxl_follower_node/freq")
        device = rospy.get_param("dxl_follower_node/device")
        baudrate = rospy.get_param("dxl_follower_node/baudrate")
        motor_list = rospy.get_param("dxl_follower_node/motor_list")
        motor_list = [eval(m) for m in motor_list.split(",")]
        main(freq, motor_list, device, baudrate)
    except rospy.ROSInterruptException:
        pass
