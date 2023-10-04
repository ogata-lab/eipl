#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import cv2
import glob
import rospy
import rosbag
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("bag_dir", type=str)
parser.add_argument("--freq", type=float, default=10)
args = parser.parse_args()


files = glob.glob(os.path.join(args.bag_dir, "*.bag"))
files.sort()
for file in files:
    print(file)
    savename = file.split(".bag")[0] + ".npz"

    # Open the rosbag file
    bag = rosbag.Bag(file)

    # Get the start and end times of the rosbag file
    start_time = bag.get_start_time()
    end_time = bag.get_end_time()

    # Get the topics in the rosbag file
    # topics = bag.get_type_and_topic_info()[1].keys()
    topics = [
        "/torobo/joint_states",
        "/torobo/head/see3cam_left/camera/color/image_repub/compressed",
        "/torobo/left_hand_controller/state",
    ]

    # Create a rospy.Time object to represent the current time
    current_time = rospy.Time.from_sec(start_time)

    joint_list = []
    finger_list = []
    image_list = []
    finger_state_list = []

    prev_finger = None
    finger_state = 0

    # Loop through the rosbag file at regular intervals (args.freq)
    freq = 1.0 / float(args.freq)
    while current_time.to_sec() < end_time:
        print(current_time.to_sec())

        # Get the messages for each topic at the current time
        for topic in topics:
            for topic_msg, msg, time in bag.read_messages(topic):
                if time >= current_time:
                    if topic == "/torobo/joint_states":
                        joint_list.append(msg.position[7:14])

                    if (
                        topic
                        == "/torobo/head/see3cam_left/camera/color/image_repub/compressed"
                    ):
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        np_img = np_img[::2, ::2]
                        image_list.append(np_img[150:470, 110:430].astype(np.uint8))

                    if topic == "/torobo/left_hand_controller/state":
                        finger = np.array(msg.desired.positions[3])
                        if prev_finger is None:
                            prev_finger = finger

                        if finger - prev_finger > 0.005 and finger_state == 0:
                            finger_state = 1
                        elif prev_finger - finger > 0.005 and finger_state == 1:
                            finger_state = 0
                        prev_finger = finger

                        finger_list.append(finger)
                        finger_state_list.append(finger_state)

                    break

        # Wait for the next interval
        current_time += rospy.Duration.from_sec(freq)
        rospy.sleep(freq)

    # Close the rosbag file
    bag.close()

    # Convert list to array
    joints = np.array(joint_list, dtype=np.float32)
    finger = np.array(finger_list, dtype=np.float32)
    finger_state = np.array(finger_state_list, dtype=np.float32)
    images = np.array(image_list, dtype=np.uint8)

    # Get shorter lenght
    shorter_length = min(len(joints), len(images), len(finger), len(finger_state))

    # Trim
    joints = joints[:shorter_length]
    finger = finger[:shorter_length]
    images = images[:shorter_length]
    finger_state = finger_state[:shorter_length]

    # Save
    np.savez(
        savename, joints=joints, finger=finger, finger_state=finger_state, images=images
    )
