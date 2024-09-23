#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import matplotlib.pylab as plt
from eipl.layer import CoordConv2d, AddCoords


add_coord = AddCoords(with_r=True)
x_im = torch.zeros(7, 12, 64, 64)
y_im = add_coord(x_im)
print("x_shape:", x_im.shape)
print("y_shape:", y_im.shape)

plt.subplot(1, 3, 1)
plt.imshow(y_im[0, -3])
plt.subplot(1, 3, 2)
plt.imshow(y_im[0, -2])
plt.subplot(1, 3, 3)
plt.imshow(y_im[0, -1])
plt.show()
