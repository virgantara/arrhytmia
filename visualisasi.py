import librosa
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'dataset/own/'

normal_sounds = glob(data_dir + '/normal' + '/*.wav')
abnormal_sounds = glob(data_dir + '/abnormal' + '/*.wav')

signal, sr = librosa.load(normal_sounds[0])
time = np.arange(0, len(signal)) / sr

fig, ax = plt.subplots()
ax.plot(time,signal)
ax.set(xlabel='Time (s)', ylabel='Ampl')


signal, sr = librosa.load(abnormal_sounds[0])
time = np.arange(0, len(signal)) / sr

fig, ax = plt.subplots()
ax.plot(time,signal)
ax.set(xlabel='Time (s)', ylabel='Ampl')


plt.show()