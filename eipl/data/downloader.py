#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import tarfile
import numpy as np
import urllib.request
from urllib.error import URLError
from eipl.utils import normalization
from .data_dict import data_dict


class Downloader:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.root_dir = os.path.join(os.path.expanduser("~"), ".eipl/", robot)

    def _download_tar_files(self, mirror_url):
        for _url in mirror_url:
            self._download(_url)

    def _check_exists(self, filepath):
        """Checks whether or not a given file path actually exists.
        Returns True if the file exists, False if it does not.

        Args:
            filepath ([string]): Path of the file to check.

        Returns:
            bool: True/False
        """
        return os.path.isfile(filepath)

    def _download(self, mirror_url):
        """Download the data if it doesn't exist already."""

        filename = os.path.splitext(os.path.basename(mirror_url))[0]
        root_dir = os.path.join(os.path.expanduser("~"), ".eipl/", self.robot)
        tar_file = os.path.join(root_dir, filename + ".tar")
        os.makedirs(self.root_dir, exist_ok=True)

        # download files
        try:
            if not self._check_exists(tar_file):
                print(f"Downloading {mirror_url}")
                urllib.request.urlretrieve(mirror_url, tar_file)

            with tarfile.open(tar_file, "r:tar") as tar:
                tar.extractall(path=self.root_dir)

        except URLError as error:
            raise RuntimeError(f"Error downloading")


class SampleDownloader(Downloader):
    """Load the sample datasets. Both AIREC and OpenManipulator object grasping data are available.

    !!! example "Example usage"

        ```py
        sample_data = SampleDownloader("airec", "grasp_bottle")
        train_img, train_joints = sample_data.load_raw_data("train")
        ```

    Arguments:
        robot (string): Name of the robot. Currently, the program supports AIREC and OpenManipulator.
        task (string): Name of experimental task. Task name differs for each robot, see data_dict.
        img_format (string): A typical image order is height, width, channel; if image_format='CHW', use transpose to reorder the channels for pytorch.
    """

    def __init__(self, robot, task, img_format="CHW"):
        super().__init__(robot=robot, task=task)
        self.robot = robot
        self.task = task
        self.img_format = img_format
        self.root_dir = os.path.join(os.path.expanduser("~"), ".eipl/", robot)
        mirror_urls = data_dict[robot][task]

        # download data
        self._download_tar_files(mirror_urls)

        # load npy data
        self.joint_bounds = self._load_bounds()

    def _load_bounds(self):
        """Download the data if it doesn't exist already.
        The joint angles' maximum and minimum values are obtained as joint_bounds.
        These values can either represent the overall maximum and minimum values in
        the dataset or the maximum and minimum values for each individual joint angle.

        Returns:
            joint_bounds (numpy.array): The min/max bounder of joints angles, expected to be 2D array
        """
        joint_bounds = np.load(os.path.join(self.root_dir, self.task, "joint_bounds.npy"))

        return joint_bounds

    def _load_data(self, data_type):
        """The function reads and returns image data and joint data of the specified type ("train" or "test").
         The order of image data (CHW/HWC) can be changed by arguments.

        Args:
            data_type ([string]): Sets whether train or test data is to be loaded.

        Returns:
            images (numpy.array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy.array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
        """
        joints = np.load(os.path.join(self.root_dir, self.task, data_type, "joints.npy"))
        images = np.load(os.path.join(self.root_dir, self.task, data_type, "images.npy"))
        if self.img_format == "CHW":
            images = images.transpose(0, 1, 4, 2, 3)

        return images, joints

    def load_raw_data(self, data_type="train"):
        """Loads data according to the specified data type.
        Args:
            data_type (Optional[string]): Sets whether train or test data is to be loaded.

        Returns:
            images (numpy.array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy.array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
        """

        return self._load_data(data_type)

    def load_norm_data(self, data_type="train", vmin=0.1, vmax=0.9):
        """Loads data with normalization to a specified range.
        Args:
            data_type (Optional[string]): Default is train
            vmin (Optional[float]): Lower limit of normalized data
            vmax (Optional[float]): Upper limit of normalized data

        Returns:
            images (numpy.array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy.array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
        """
        images_raw, joints_raw = self._load_data(data_type)
        images = normalization(images_raw.astype(np.float32), (0.0, 255.0), (vmin, vmax))
        joints = normalization(joints_raw.astype(np.float32), self.joint_bounds, (vmin, vmax))

        return images, joints


class WeightDownloader(Downloader):
    """Download the pretrained weight.

    !!! example "Example usage"

        ```py
        WeightDownloader("airec", "grasp_bottle")
        ```

    Arguments:
        robot (string): Name of the robot. Currently, the program supports AIREC and OpenManipulator.
        task (string): Name of experimental task. Task name differs for each robot, see data_dict.
    """

    def __init__(self, robot, task):
        super().__init__(robot=robot, task=task)
        self.robot = robot
        self.task = task
        self.root_dir = os.path.join(os.path.expanduser("~"), ".eipl/", robot)

        # download data
        self._download_tar_files(data_dict[robot][task])
