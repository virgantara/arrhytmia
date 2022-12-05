from helper import parse_audio_files, write_our_features, write_yadav_features, write_yaseen_features, \
    write_ghosh_features, write_mei_features
import glob
import numpy as np
import pywt
import librosa
import matplotlib.pyplot as plt
from scipy.stats import entropy

# y, sr = librosa.load('dataset/Yaseen/AS/New_AS_001.wav', sr=44100)
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=19, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))
# mfcc_mean = np.mean(mfcc, axis=1)
#
# coeffs = pywt.wavedec(data=y, wavelet='db1', level=3)
# cA2, cA1, cD2, cD1 = coeffs
# cA2_mean = np.mean(cA2)
# cA2_std = np.std(cA2)
# cA2_max = np.max(cA2)
#
# cA1_mean = np.mean(cA1)
# cA1_std = np.std(cA1)
# cA1_max = np.max(cA1)
#
# cD2_mean = np.mean(cD2)
# cD2_std = np.std(cD2)
# cD2_max = np.max(cD2)
#
# cD1_mean = np.mean(cD1)
# cD1_std = np.std(cD1)
# cD1_max = np.max(cD1)


# plt.plot(cA2)
# plt.show()

# dataset_dir = 'dataset/anomali'
# LIMIT = 100

# write_mei_features('dataset/Yaseen/AS','AS')
# write_mei_features('dataset/Yaseen/N','N')
# write_mei_features('dataset/Yaseen/MR','MR')
# write_mei_features('dataset/Yaseen/MS','MS')
# write_mei_features('dataset/Yaseen/MVP','MVP')



# write_yadav_features(dataset_dir+'/normal',label='normal')
# write_yadav_features(dataset_dir+'/abnormal',label='abnormal')

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