#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch

print("torch version:", torch.__version__)

if torch.cuda.is_available():
    print("cuda is available")
    print("device_count: ", torch.cuda.device_count())
    print("device name: ", torch.cuda.get_device_name())
else:
    print("cuda is not avaiable")
