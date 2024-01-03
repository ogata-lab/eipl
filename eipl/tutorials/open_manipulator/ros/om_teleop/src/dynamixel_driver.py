#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

from dynamixel_utils import *
from dynamixel_sdk import *

# ROS
import rospy
from sensor_msgs.msg import JointState


class DynamixelDriver:
    def __init__(self, motor_list=[0, 1], device="/dev/ttyUSB0", baudrate=1000000):
        self.motor_list = motor_list
        self.baudrate = baudrate

        # Initialize PortHandler/PacketHandler Structs
        self.portHandler = PortHandler(device)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.serial_open()
        self.set_gripper_mode()

    def serial_open(self):
        if self.portHandler.openPort():
            rospy.loginfo("Succeeded to open the port")
        else:
            rospy.loginfo("Failed to open the port")
            self.clean_shutdown()

        # Set port baudrate
        if self.portHandler.setBaudRate(self.baudrate):
            rospy.loginfo("Succeeded to change the baudrate")
        else:
            rospy.loginfo("Failed to change the baudrate")
            self.clean_shutdown()

    def set_gripper_mode(self):
        motor_id = self.motor_list[-1]

        # change to current-based position control mode
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler,
            motor_id,
            ADDR_OPERATING_MODE,
            CURRENT_POSITION_CONTROL_MODE,
        )
        if dxl_comm_result != COMM_SUCCESS:
            rospy.loginfo("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            rospy.loginfo("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            rospy.loginfo("Dynamixel#%d has been successfully connected" % motor_id)

        # set goal current
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, ADDR_GOAL_CURRENT, 50
        )
        if dxl_comm_result != COMM_SUCCESS:
            rospy.loginfo("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            rospy.loginfo("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            rospy.loginfo("Dynamixel#%d has been successfully connected" % motor_id)

    def torque_on(self):
        for motor_id in self.motor_list:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
            )
            if dxl_comm_result != COMM_SUCCESS:
                rospy.loginfo("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                rospy.loginfo("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                rospy.loginfo("Dynamixel#%d has been successfully connected" % motor_id)

    def torque_off(self):
        for motor_id in self.motor_list:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
            )
            if dxl_comm_result != COMM_SUCCESS:
                rospy.loginfo("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                rospy.loginfo("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def set_joint_raw(self, target_angle):
        for index, motor_id in enumerate(self.motor_list):
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler, motor_id, ADDR_GOAL_POSITION, target_angle[index]
            )
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def set_joint_radian(self, target_angle_radian):
        target_angle_raw = radian2value(target_angle_radian)
        self.set_joint_raw(target_angle_raw)

    def get_joint_raw(self):
        joint_list = []
        for motor_id in self.motor_list:
            (
                dxl_present_position,
                dxl_comm_result,
                dxl_error,
            ) = self.packetHandler.read4ByteTxRx(
                self.portHandler, motor_id, ADDR_PRESENT_POSITION
            )
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            joint_list.append(dxl_present_position)

        return joint_list

    def get_joint_degree(self):
        return value2degree(self.get_joint_raw())

    def get_joint_radian(self):
        return value2radian(self.get_joint_raw())

    def serial_close(self):
        self.portHandler.closePort()
        rospy.loginfo("Close serial port")

    def clean_shutdown(self):
        self.torque_off()
        self.serial_close()
        exit()


if __name__ == "__main__":
    try:
        rospy.init_node("dynamixel_driver_node", anonymous=True)
        dxl_driver = DynamixelDriver(motor_list=[10])
        r = rospy.Rate(250)
        while not rospy.is_shutdown():
            start_time = time.time()

            joint_angle = dxl_driver.get_joint_angle()

            r.sleep()
            freq = 1.0 / (time.time() - start_time)
            print(freq, joint_angle)

        dxl_driver.serial_close()
    except rospy.ROSInterruptException:
        pass
