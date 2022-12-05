import librosa, librosa.display
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from scipy import signal
import soundfile as sf
from helper import filter_signal

data_dir = 'dataset/own/'

normal_sounds = glob(data_dir + '/normal' + '/*.wav')
abnormal_sounds = glob(data_dir + '/abnormal' + '/*.wav')

y, sr = librosa.load(normal_sounds[15])
time = np.arange(0, len(y)) / sr

filtered_signal = filter_signal(data=y, cut_off=5512, sr=sr)

S = np.abs(librosa.stft(filtered_signal))
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.show()

# sf.write('normal_filtered.wav',filtered_signal, sr, subtype='PCM_24')
