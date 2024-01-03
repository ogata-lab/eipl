#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import numpy as np
from eipl.utils import resize_img, normalization, cos_interpolation, calc_minmax


def load_data(index, data_dir, train):
    image_list = []
    joint_list = []
    pose_list = []

    num_list = [1, 2, 3, 4, 5] if train else [6, 7]
    for h in index:
        for i in num_list:
            filename = data_dir.format(h, i)
            print(filename)
            data = np.load(filename, allow_pickle=True)

            images = data["images"]
            images = np.array(images)[:, ::-1]
            images = resize_img(images, (64, 64))

            _poses = data["poses"]
            _joints = data["joints"]
            _gripper = data["gripper"][:, 0]
            _gripper = normalization(_gripper, (-1, 1), (0, 1))
            _gripper = cos_interpolation(_gripper, 10, expand_dims=True)
            poses = np.concatenate((_poses, _gripper), axis=-1)
            joints = np.concatenate((_joints, _gripper), axis=-1)

            joint_list.append(joints)
            image_list.append(images)
            pose_list.append(poses)

    joints = np.array(joint_list)
    images = np.array(image_list)
    poses = np.array(pose_list)

    return images, joints, poses


if __name__ == "__main__":
    # train position: collect 7 data/position (5 for train, 2 for test)
    # test  position: collect 2 data/position (all for test)
    #            Pos1 Pos2  Pos3  Pos4 Pos5 Pos6 Pos7 Pos8 Pos9
    # pos_train: -0.2       -0.1       0.0       0.1       0.2
    # pos_test:  -0.2 -0.15 -0.1 -0.05 0.0  0.05 0.1  0.15 0.2

    data_dir = "./data/raw_data/Pos{}_{}/state_resave.npz"

    # load train data
    train_index = [1, 3, 5, 7, 9]
    train_images, train_joints, train_poses = load_data(
        train_index, data_dir, train=True
    )
    np.save("./data/train/images.npy", train_images.astype(np.uint8))
    np.save("./data/train/joints.npy", train_joints.astype(np.float32))
    np.save("./data/train/poses.npy", train_poses.astype(np.float32))

    test_index = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_images, test_joints, test_poses = load_data(test_index, data_dir, train=False)
    np.save("./data/test/images.npy", test_images.astype(np.uint8))
    np.save("./data/test/joints.npy", test_joints.astype(np.float32))
    np.save("./data/test/poses.npy", test_poses.astype(np.float32))

    # save bounds
    poses = np.concatenate((train_poses, test_poses), axis=0)
    joints = np.concatenate((train_joints, test_joints), axis=0)
    pose_bounds = calc_minmax(poses)
    joint_bounds = calc_minmax(joints)
    np.save("./data/pose_bounds.npy", pose_bounds)
    np.save("./data/joint_bounds.npy", joint_bounds)
