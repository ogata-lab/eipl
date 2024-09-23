#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import numpy as np
import matplotlib
import matplotlib.pylab as plt
from eipl.utils import get_mean_minmax, get_bounds, normalization

matplotlib.use("tkagg")

vmin = 0.1
vmax = 0.9

x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.sin(x + 1.0)
y3 = np.sin(x + 2.0)

data = np.array([y1 * 0.1, y2 * 0.5, y3 * 0.8]).T
# Change data shape from [seq_len, dim] to [N, seq_len, dim]
data = np.expand_dims(data, axis=0)
data_mean, data_min, data_max = get_mean_minmax(data)
bounds = get_bounds(data_mean, data_min, data_max, clip=0.2, vmin=vmin, vmax=vmax)
norm_data = normalization(data - bounds[0], bounds[1:], (vmin, vmax))
denorm_data = normalization(norm_data, (vmin, vmax), bounds[1:]) + bounds[0]

plt.figure()
plt.plot(data[0])
plt.title("original data")

plt.figure()
plt.plot(norm_data[0])
plt.title("normalized data")

plt.figure()
plt.plot(denorm_data[0])
plt.title("deormalized data")

plt.show()
