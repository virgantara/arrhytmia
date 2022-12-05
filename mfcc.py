import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

signal, sr = librosa.load('dataset/usg/Abnormal/f13.wav')

mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
print(mfccs.shape)

plt.figure(figsize=(25,10))
librosa.display.specshow(mfccs,x_axis='time',sr=sr)

plt.colorbar(format='%+2f')
plt.show()