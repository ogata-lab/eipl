#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import numpy as np
import matplotlib.pylab as plt
from eipl.layer import GridMask


# single image with channel last
img = np.ones((64, 64, 3))
mask = GridMask(channel_first=False)
out = mask(img, debug=True)
plt.figure(dpi=60)
plt.imshow(out)

# single image with channel first
img = np.ones((3, 64, 64))
mask = GridMask(channel_first=True)
out = mask(img, debug=True)
out = out.transpose(1, 2, 0)
plt.figure(dpi=60)
plt.imshow(out)

# multi images with channel last
img = np.ones((4, 64, 64, 3))
mask = GridMask(channel_first=False)
out = mask(img, debug=True)

plt.figure(dpi=60)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(out[i])

# multi images with channel first
img = np.ones((4, 3, 64, 64))
mask = GridMask(channel_first=True)
out = mask(img, debug=True)
out = out.transpose(0, 2, 3, 1)

plt.figure(dpi=60)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(out[i])
plt.show()
