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
parser.add_argument("--bag_dir", type=str, default="./bag/")
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
    topics = ["/follower/joint_states", "/camera/color/image_raw/compressed"]

    # Create a rospy.Time object to represent the current time
    current_time = rospy.Time.from_sec(start_time)

    joint_list = []
    image_list = []

    prev_gripper = None
    gripper_state = 0

    # Loop through the rosbag file at regular intervals (args.freq)
    freq = 1.0 / float(args.freq)
    while current_time.to_sec() < end_time:
        # Get the messages for each topic at the current time
        for topic in topics:
            for topic_msg, msg, time in bag.read_messages(topic, start_time=current_time):
                if time >= current_time:
                    if topic == "/camera/color/image_raw/compressed":
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        np_img = np_img[::2, ::2]
                        image_list.append(np_img.astype(np.uint8))

                    if topic == "/follower/joint_states":
                        joint_list.append(msg.position)
                    break

        # Wait for the next interval
        current_time += rospy.Duration.from_sec(freq)
    
    # Close the rosbag file
    bag.close()

    # Convert list to array
    joints = np.array(joint_list, dtype=np.float32)
    images = np.array(image_list, dtype=np.uint8)

    # # Get shorter lenght
    shorter_length = min(len(joints), len(images))

    # # Trim
    joints = joints[:shorter_length]
    images = images[:shorter_length]

    # # Save
    np.savez(savename, joints=joints, images=images)
