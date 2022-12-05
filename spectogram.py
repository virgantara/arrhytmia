import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

NUM_OF_DATA = 3  # int(args["data"])


if not os.path.exists('spectogram/mfcc/'):
    os.mkdir('spectogram/mfcc/')


dirname = 'spectogram/mfcc/normal'

if not os.path.exists(dirname):
    os.mkdir(dirname)

dataset_path = 'dataset/own/normal'
counter = 0
for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    for f in filenames:
        file_path = os.path.join(dirpath, f)
        signal, sr = librosa.load(file_path, sr=16000)
        print(f)
        parser = []
        parser.append(-1)
        splitter = int(len(signal) / NUM_OF_DATA)

        n_fft = 2048
        hop_length = 512

        for i in range(1, NUM_OF_DATA + 1):
            parser.append(parser[i - 1] + splitter)
            counter += 1
            # STFT
            # stft = librosa.core.stft(signal[parser[i - 1] + 1:parser[i]], hop_length=hop_length, n_fft=n_fft)
            # spectogram = np.abs(stft)

            # log_spectogram = librosa.amplitude_to_db(spectogram)
            #
            # librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)

            # MFCCs
            mfcc = librosa.feature.mfcc(y=signal[parser[i-1]+1:parser[i]], hop_length=hop_length, n_mfcc=13)
            librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length,x_axis='time')

            plt.savefig(dirname+'/data_' + str(counter) + '.png')
        # print(parser)
