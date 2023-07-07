#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm
import matplotlib.pylab as plt


def normalization(data, indataRange, outdataRange):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. indataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indataRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array
    """
    data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
    data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    return data


def deprocess_img(data, vmin=-0.9, vmax=0.9):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        vmin (float):  Minimum value of input data
        vmax (float):  Maximum value of input data
    Return:
        data (np.array with np.uint8): Normalized data array from 0 to 255.
    """
    data[np.where(data < vmin)] = vmin
    data[np.where(data > vmax)] = vmax
    return normalization(data, [vmin, vmax], [0, 255]).astype(np.uint8)


def tensor2numpy(x):
    """
    Convert tensor to numpy array.
    """
    if x.device.type == "cpu":
        return x.detach().numpy()
    else:
        return x.cpu().detach().numpy()


def resize_img(img, size=(64, 64), reshape_flag=True):
    """
    Convert tensor to numpy array.
    """
    if len(img.shape) == 5:
        N, T, W, H, C = img.shape
        img = img.reshape((-1,) + img.shape[2:])
    else:
        reshape_flag = False

    imgs = []
    for i in range(len(img)):
        imgs.append(cv2.resize(img[i], size))

    imgs = np.array(imgs)
    if reshape_flag:
        imgs = imgs.reshape(N, T, size[1], size[0], 3)
    return imgs


def plt_img(img, key=None, title=None, dtype=None):
    """
    Convert tensor to numpy array.
    """
    width, height = img.shape

    if dtype is not None:
        plt.imshow(img.astype(dtype))
    else:
        plt.imshow(img)

    if key is not None:
        plt.plot(key[0] * width, key[1] * height, "ro", markersize=3)

    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.title(title)
    plt.tick_params(length=0)


def calc_minmax(_data):
    data = _data.reshape(-1, _data.shape[-1])
    data_minmax = np.array([np.min(data, 0), np.max(data, 0)])
    return data_minmax


def list_to_numpy(data_list, max_N):
    dtype = data_list[0].dtype
    array = np.ones(
        (
            len(data_list),
            max_N,
        )
        + data_list[0].shape[1:],
        dtype,
    )

    for i, data in enumerate(data_list):
        N = len(data)
        array[i, :N] = data[:N].astype(dtype)
        array[i, N:] = array[i, N:] * data[-1].astype(dtype)

    return array


def cos_interpolation(data, step=20):
    data = data.copy()
    points = np.diff(data)

    for i, p in enumerate(points):
        if p == 1:
            t = np.linspace(0.0, 1.0, step * 2)
        elif p == -1:
            t = np.linspace(1.0, 0.0, step * 2)
        else:
            continue

        x_latent = (1 - np.cos(t * np.pi)) / 2
        data[i - step + 1 : i + step + 1] = x_latent

    return np.expand_dims(data, axis=-1)


def get_lissajous(total_step, num_cycle, x_mag, y_mag, delta, dtype=np.float32):
    """
    Function to generate a Lissajous curve
    Reference URL: http://www.ne.jp/asahi/tokyo/nkgw/www_2/gakusyu/rikigaku/Lissajous/Lissajous_kaisetu/Lissajous_kaisetu_1.html
    Args:
        total_step (int): Sequence length of Lissajous curve.
        num_cycle  (int): Iteration of the Lissajous curve.
        x_mag    (float): Angular frequency of the x direction.
        y_mag    (float): Angular frequency of the y direction.
        delta    (float): Initial phase of the y direction
    Return:
        data (np.array): Array of Lissajous curves. data shape is [total_step, 2]
    """
    t = np.linspace(0, 2.0 * np.pi * num_cycle, total_step)
    x = np.cos(t * x_mag)
    y = np.cos(t * y_mag + delta)

    return np.c_[x, y].astype(dtype)


def get_lissajous_movie(
    total_step, num_cycle, x_mag, y_mag, delta, imsize, circle_r, color, vmin=-0.9, vmax=0.9
):
    """
    Function to generate a Lissajous curve with movie
    Args:
        total_step (int): Sequence length of Lissajous curve.
        num_cycle  (int): Iteration of the Lissajous curve.
        x_mag    (float): Angular frequency of the x direction.
        y_mag    (float): Angular frequency of the y direction.
        delta    (float): Initial phase of the y direction
        imsize     (int): Pixel size of the movie
        circle_r   (int): Radius of the circle moving in the movie.
        color     (list): Color of the circle. Specify color in RGB list, e.g. red is [255,0,0].
        vmin     (float): Minimum value of output data
        vmax     (float): Maximum value of output data
    Return:
        data (np.array): Array of movie and curve. movie shape is [total_step, imsize, imsize, 3], curve shape is [total_step, 2].
    """

    xy = getLissajous(total_step, num_cycle, x_mag, y_mag, delta)
    x, y = np.split(xy, indices_or_sections=2, axis=-1)

    _color = tuple((np.array(color)).astype(np.uint8))

    imgs = []
    for _t in range(total_step):
        # xy position in the image
        _x = (x[_t] * (imsize * 0.4)) + imsize / 2
        _y = (y[_t] * (imsize * 0.4)) + imsize / 2
        img = Image.new("RGB", (imsize, imsize), "white")
        draw = ImageDraw.Draw(img)
        # Draws a circle with a specified radius
        draw.ellipse((_x - circle_r, _y - circle_r, _x + circle_r, _y + circle_r), fill=_color)
        imgs.append(np.expand_dims(np.asarray(img), 0))
    imgs = np.vstack(imgs)

    ### normalization
    imgs = normalization(imgs.astype(np.float32), [0, 255], [vmin, vmax])
    seq = normalization(np.c_[x, y].astype(np.float32), [-1.0, 1.0], [vmin, vmax])
    return imgs, seq


def get_feature_map(im_size=64, channels=3, size=1000):
    """
    Returns inference, using input x and the last step's hidden state.
    Args:
        im_size (int):
        channels (int):
        size (int):

    Outputs:
        featmap (np_array): batch_size, channels, width, height = x.shape
    """
    feat_map = []
    for _ in range(channels):
        mean1 = [np.random.randint(im_size), np.random.randint(im_size)]
        cov1 = [[10, 0], [0, 20]]
        mean2 = [np.random.randint(im_size), np.random.randint(im_size)]
        cov2 = [[10, 0], [0, 5]]

        x, y = np.vstack(
            (
                np.random.multivariate_normal(mean1, cov1, size),
                np.random.multivariate_normal(mean2, cov2, size),
            )
        ).T
        _feat_map, _, _, _ = plt.hist2d(x, y, bins=im_size, cmap=cm.gray)
        plt.close()
        feat_map.append(_feat_map)

    return np.array([feat_map], dtype=np.float32)
