#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import matplotlib.pylab as plt
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax
from eipl.utils import tensor2numpy, plt_img, get_feature_map


channels = 4
im_size = 64
temperature = 1e-4
heatmap_size = 0.05

# Generate feature_map
in_features = get_feature_map(im_size=im_size, channels=channels)
print("in_features shape: ", in_features.shape)

# Apply spetial softmax, get keypoints and attention map
ssm = SpatialSoftmax(
    width=im_size, height=im_size, temperature=temperature, normalized=True
)
keypoints, att_map = ssm(torch.tensor(in_features))
print("keypoints shape: ", keypoints.shape)

# Generate heatmap from keypoints
issm = InverseSpatialSoftmax(
    width=im_size, height=im_size, heatmap_size=heatmap_size, normalized=True
)
out_features = issm(keypoints)
out_features = tensor2numpy(out_features)
print("out_features shape: ", out_features.shape)

plt.figure(dpi=60)
# feature map
for i in range(1, channels + 1):
    plt.subplot(2, channels, i)
    plt_img(
        in_features[0, i - 1], key=keypoints[0, i - 1], title="feature map {}".format(i)
    )

# plot heatmap
for i in range(1, channels + 1):
    plt.subplot(2, channels, channels + i)
    plt_img(out_features[0, i - 1], title="heatmap map {}".format(i))

plt.tight_layout()
# plt.savefig("spatial_softmax.png")
plt.show()
