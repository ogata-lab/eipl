#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
from torch.utils.data import Dataset, DataLoader
try:
    from torchvision.transforms import v2 as transforms
except ImportError:
    from torchvision import transforms


class ImageDataset(Dataset):
    """
    This class is used to train models that deal only with images, such as autoencoders.
    Data augmentation is applied to the given image data by adding lightning, contrast, horizontal and vertical shift, and gaussian noise.

    Arguments:
        data (numpy.array): Set the data type (train/test). If the last three dimensions are HWC or CHW, `data` allows any number of dimensions.
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, data, stdev=None):
        """
        Reshapes and transforms the data.

        Arguments:
            data (numpy.array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            stdev (float, optional): The standard deviation for the normal distribution to generate gaussian noise.
        """

        self.stdev = stdev
        _image_flatten = data.reshape(((-1,) + data.shape[-3:]))
        self.image_flatten = torch.from_numpy(_image_flatten).float()

        self.transform_affine = transforms.Compose(
            [
                transforms.RandomAffine(degrees=(0, 0), translate=(0.15, 0.15)),
                transforms.RandomAutocontrast(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        self.transform_noise = transforms.Compose(
            [
                transforms.ColorJitter(
                    contrast=[0.6, 1.4], brightness=0.4, saturation=[0.6, 1.4], hue=0.04
                )
            ]
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            length (int): The length of the dataset.
        """

        return len(self.image_flatten)

    def __getitem__(self, idx):
        """
        Extracts a single image from the dataset and returns two images: the original image and the image with noise added.

        Args:
            idx (int): The index of the element.

        Returns:
            image_list (list): A list containing the transformed and noise added image (x_img) and the affine transformed image (y_img).
        """
        img = self.image_flatten[idx]

        if self.stdev is not None:
            y_img = self.transform_affine(img)
            x_img = self.transform_noise(y_img) + torch.normal(
                mean=0, std=self.stdev, size=y_img.shape
            )
        else:
            y_img = img
            x_img = img

        return [x_img, y_img]


class MultimodalDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., images, joints), such as CNNRNN/SARNN.

    Args:
        images (numpy array): Set of images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, images, joints, device, stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the images, joints, and transformation.

        Args:
            images (numpy array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.device = device
        self.stdev = stdev
        self.images = torch.Tensor(images).to(self.device)
        self.joints = torch.Tensor(joints).to(self.device)
        self.transform = transforms.ColorJitter(
            contrast=[0.6, 1.4], brightness=0.4, saturation=[0.6, 1.4], hue=0.04
        ).to(self.device)

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of images and joints at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and joint (x_img, x_joint) and the original image and joint (y_img, y_joint).
        """
        x_img = self.images[idx]
        x_joint = self.joints[idx]
        y_img = self.images[idx]
        y_joint = self.joints[idx]

        if self.stdev is not None:
            x_img = self.transform(y_img)
            x_img = x_img + torch.normal(mean=0, std=0.02, size=x_img.shape, device=self.device)
            x_joint = y_joint + torch.normal(mean=0, std=self.stdev, size=y_joint.shape, device=self.device)

        return [[x_img, x_joint], [y_img, y_joint]]


class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
