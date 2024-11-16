#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import numpy as np
import matplotlib
import matplotlib.pylab as plt
from eipl.utils import cos_interpolation

matplotlib.use("tkagg")

data = np.zeros(120)
data[30:70] = 1
smoothed_data = cos_interpolation(data, step=15)

plt.plot(data)
plt.plot(smoothed_data)
plt.show()
