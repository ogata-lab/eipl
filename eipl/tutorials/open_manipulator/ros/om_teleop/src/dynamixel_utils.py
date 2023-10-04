#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys, tty, termios
import numpy as np

# Control table address (need to change)
ADDR_OPERATING_MODE = 1  # Data Byte Length
CURRENT_POSITION_CONTROL_MODE = 5
ADDR_GOAL_CURRENT = 102
ADDR_TORQUE_ENABLE = 64
ADDR_LED_RED = 65
LEN_LED_RED = 1  # Data Byte Length
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4  # Data Byte Length
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4  # Data Byte Length
DXL_MINIMUM_POSITION_VALUE = 0  # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE = (
    4095  # Refer to the Maximum Position Limit of product eManual
)
TORQUE_ENABLE = 1  # Value for enabling the torque
TORQUE_DISABLE = 0  # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20  # Dynamixel moving status threshold
PROTOCOL_VERSION = 2.0


## getch
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)


def getch():
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


## value2radian
def value2degree(data):
    """convert dynamixel value to radian"""
    return np.array(data, dtype=np.float32) * 360.0 / 4095.0


def value2radian(data):
    """convert dynamixel value to degree"""
    return (np.array(data, dtype=np.float32) - 2047.5) / 2047.5 * np.pi


def degree2value(data):
    """convert dynamixel value to radian"""
    return np.array(data, dtype=np.float32) * 4095.0 / 360.0


def radian2value(data):
    """convert dynamixel value to degree"""
    val = np.array(data, dtype=np.float32) / np.pi * 2047.5 + 2047.5
    return val.astype(np.int32)


def normalization(dselfata, indataRange, outdataRange):
    data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
    data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    return data
