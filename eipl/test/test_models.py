#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

from eipl.layer import MTRNNCell
from eipl.model import BasicLSTM, BasicMTRNN
from eipl.model import BasicCAE, CAE
from eipl.model import BasicCAEBN, CAEBN
from eipl.model import CNNRNN, CNNRNNLN, SARNN
from torchinfo import summary


batch_size = 50
input_dim = 12

### MTRNNCell
print("# MTRNNCell")
model = MTRNNCell(input_dim=12, fast_dim=50, slow_dim=5, fast_tau=2, slow_tau=12)
summary(model, input_size=(batch_size, input_dim))

print("# BasicMTRNN")
model = BasicMTRNN(in_dim=12, fast_dim=30, slow_dim=5, fast_tau=2, slow_tau=12, activation="tanh")
summary(model, input_dim=(batch_size, input_dim))

print("# BasicLSTM")
model = BasicLSTM(in_dim=12, rec_dim=50, out_dim=10, activation="tanh")
summary(model, input_dim=(batch_size, input_dim))

print("BasicCAE")
model = BasicCAE()
summary(model, input_size=(batch_size, 3, 128, 128))

print("CAE")
model = CAE()
summary(model, input_size=(batch_size, 3, 128, 128))

print("BasicCAEBN")
model = BasicCAEBN(feat_dim=10)
summary(model, input_size=(batch_size, 3, 128, 128))

print("CAEBN")
model = CAEBN(feat_dim=30)
summary(model, input_size=(batch_size, 3, 128, 128))


print("CNNRNN")
model = CNNRNN(rec_dim=50, joint_dim=8, feat_dim=10)
summary(model, input_size=[(batch_size, 3, 128, 128), (batch_size, 8)])

print("CNNRNNLN")
model = CNNRNNLN(rec_dim=50, joint_dim=8, feat_dim=10)
summary(model, input_size=[(batch_size, 3, 128, 128), (batch_size, 8)])

print("SARNN")
model = SARNN(rec_dim=50, k_dim=5, joint_dim=8)
summary(model, input_size=[(batch_size, 3, 128, 128), (batch_size, 8)])
