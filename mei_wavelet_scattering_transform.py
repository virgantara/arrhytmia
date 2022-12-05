from kymatio.numpy import Scattering1D
from kymatio.datasets import fetch_fsdd
import numpy as np
import os
import scipy.io.wavfile
import matplotlib.pyplot as plt

file_path = 'dataset/Yaseen/AS/New_AS_001.wav'


sr, x = scipy.io.wavfile.read(file_path)
x = x / np.max(np.abs(x))

T = x.shape[-1]
# print(2**13)
J = 8
Q = 12
scattering = Scattering1D(J, T, Q)
Sx = scattering(x)
meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)
print("0-ord shape", Sx[order0].shape)
print("1-ord shape", Sx[order1].shape)
print("2-ord shape", Sx[order2].shape)
feat = np.vstack((Sx[order0], Sx[order1], Sx[order2]))
feat = feat[:,0:10]
feat = np.mean(feat,axis=0)
print(feat.shape)

#
# plt.figure(figsize=(8, 8))
# plt.subplot(3, 1, 1)
# plt.plot(Sx[order0][0])

# plt.title('Zeroth-order scattering')
#
# plt.subplot(3, 1, 2)
# plt.imshow(Sx[order1], aspect='auto')

# plt.title('First-order scattering')
#
# plt.subplot(3, 1, 3)
# plt.imshow(Sx[order2], aspect='auto')
# print("2-ord shape", Sx[order2].shape)
# plt.title('Second-order scattering')
#
# plt.tight_layout()
# plt.show()