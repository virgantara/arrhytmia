import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from scipy import signal
import numpy as np
import librosa
import librosa.display
import wave
import random as rn
import tensorflow as tf

from keras.utils import to_categorical
import seaborn as sns

features = []
for fn in glob.glob(os.path.join(dataset_dir+'/normal',"*.wav")):
    print(fn)
    y, sr = librosa.load(fn)
    mfcc = librosa.feature.mfcc(y=y)
    padded_mfcc = pad2d(mfcc, 40)
    expanded = np.expand_dims(padded_mfcc, -1)
    print(expanded.shape)
    # features.append(padded_mfcc)
    # features.append(mfcc_extract(fn))
# write_mfcc_features(dataset_dir+'/normal',label='normal')
# write_mfcc_features(dataset_dir+'/abnormal',label='abnormal')

# train_cat_dirs = glob.glob(dataset_dir+'/*')
# train_cat = []
# for cat_dir in train_cat_dirs:
#     tmp = cat_dir.split('/')[-1]
#     train_cat.append(tmp)
# print('the number of kinds:', len(train_cat))
#
# class_num = len(train_cat)
# features, labels = parse_audio_files(dataset_dir, train_cat, LIMIT)
#
# X = np.array(features[0].T)
# y = np.array(labels)
# print(X.shape)
# print(y.shape)