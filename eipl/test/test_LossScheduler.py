#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import numpy as np
import matplotlib.pylab as plt
from eipl.utils import LossScheduler


ax = []
ax.append(plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2))
ax.append(plt.subplot2grid((2, 6), (0, 2), colspan=2))
ax.append(plt.subplot2grid((2, 6), (0, 4), colspan=2))
ax.append(plt.subplot2grid((2, 6), (1, 1), colspan=2))
ax.append(plt.subplot2grid((2, 6), (1, 3), colspan=2))

for i, curve_name in enumerate(
    ["linear", "s", "inverse_s", "deceleration", "acceleration"]
):
    scheduler = LossScheduler(decay_end=100, curve_name=curve_name)
    loss_weight_list = []
    for _ in range(150):
        loss_weight_list.append(scheduler(loss_weight=0.1))

    ax[i].plot(loss_weight_list)
    ax[i].set_title(curve_name)
    ax[i].grid()

plt.tight_layout()
# plt.savefig("./output/loss_scheduler.png", dpi=60)
plt.show()
