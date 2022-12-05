import gc
import pickle
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, \
    GlobalAveragePooling1D, \
    concatenate, BatchNormalization, Flatten
from numpy import random
import librosa
import numpy as np
import glob
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

input_length = 10000 * 2

batch_size = 32


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5


def load_audio_file(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=16000)[0]  # , sr=16000
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        max_offset = input_length - len(data)
        offset = np.random.randint(max_offset)
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = audio_norm(data)
    return data


train_files = glob.glob("dataset/YaseenDeepLearning/audio_train/*.wav")
test_files = glob.glob("dataset/YaseenDeepLearning/audio_test/*.wav")
train_labels = pd.read_csv("yaseen_dl_Train.csv")
test_labels = pd.read_csv("yaseen_dl_Test.csv")

file_to_label = {"dataset/YaseenDeepLearning/audio_train/" + k: v for k, v in
                 zip(train_labels.fname.values, train_labels.label.values)}
list_labels = sorted(list(set(train_labels.label.values)))

test_file_to_label = {"dataset/YaseenDeepLearning/audio_test/" + k: v for k, v in
                      zip(test_labels.fname.values, test_labels.label.values)}
test_list_labels = sorted(list(set(test_labels.label.values)))

label_to_int = {k: v for v, k in enumerate(list_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}
file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}

test_label_to_int = {k: v for v, k in enumerate(test_list_labels)}
test_int_to_label = {v: k for k, v in test_label_to_int.items()}
test_file_to_int = {k: test_label_to_int[v] for k, v in test_file_to_label.items()}


def get_model():
    nclass = len(list_labels)
    inp = Input(shape=(input_length, 1))
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid", strides=2)(inp)
    img_1 = BatchNormalization()(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid", strides=2)(img_1)
    img_1 = BatchNormalization()(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid", strides=2)(img_1)
    img_1 = BatchNormalization()(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid", strides=2)(img_1)
    img_1 = BatchNormalization()(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid", strides=2)(img_1)
    img_1 = BatchNormalization()(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid", strides=3)(img_1)
    img_1 = BatchNormalization()(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid", strides=2)(img_1)
    img_1 = BatchNormalization()(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.15)(img_1)
    img_1 = Flatten()(img_1)

    dense_1 = Dense(128, activation=activations.relu)(img_1)
    dense_1 = Dropout(rate=0.3)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.RMSprop(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# In[14]:


def train_generator(list_files, batch_size=batch_size):
    while True:
        shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:, :, np.newaxis]
            batch_labels = [file_to_int[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)

            yield batch_data, batch_labels


def test_generator(list_files, batch_size=batch_size):
    while True:
        shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:, :, np.newaxis]
            batch_labels = [test_file_to_int[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)

            yield batch_data, batch_labels


tr_files, val_files = train_test_split(sorted(train_files), test_size=0.1, random_state=42)
# print(tr_files)
# for batch_files in chunker(tr_files, size=batch_size):
#     batch_data = [load_audio_file(fpath) for fpath in batch_files]
#     batch_data = np.array(batch_data)[:, :, np.newaxis]
#     batch_labels = [file_to_int[fpath] for fpath in batch_files]
#     batch_labels = np.array(batch_labels)
#     print(batch_files)


# model = get_model()

# model.load_weights("baseline_cnn.h5")
#
# history = model.fit(train_generator(tr_files), steps_per_epoch=len(tr_files)//batch_size, epochs=50,
#                    validation_data=train_generator(val_files), validation_steps=len(val_files)//batch_size,max_queue_size=20)

# model.save_weights("baseline_cnn.h5")
model = load_model('BaghelDNN.h5')

# model.save("BaghelDNN.h5")

# # convert the history.history dict to a pandas DataFrame:
# hist_df = pd.DataFrame(history.history)

# # or save to csv:
# hist_csv_file = 'baghel_history.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

# print(np.array(tr_files).shape, np.array(val_files).shape)
X_test = train_generator(val_files)
y_pred = model.predict(X_test)
print(y_pred)
# yt = y_test.argmax(axis=1)
# yp = y_pred.argmax(axis=1)
# matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


# print(y_test.shape, y_pred.shape)
# print(multilabel_confusion_matrix(y_test, y_pred))
# yhat_classes = y_pred

# # accuracy: (tp + tn) / (p + n)
# accuracy = accuracy_score(yt, yp)
# print('Accuracy: %f' % accuracy)
# # # precision tp / (tp + fp)
# precision = precision_score(yt, yp,average='weighted')
# print('Precision: %f' % precision)
# # recall: tp / (tp + fn)
# recall = recall_score(yt, yp,average='weighted')
# print('Recall: %f' % recall)
# # f1: 2 tp / (2 tp + fp + fn)
# f1 = f1_score(yt, yp,average='weighted')
# print('F1 score: %f' % f1)

# # # kappa
# kappa = cohen_kappa_score(yt, yp)
# print('Cohens kappa: %f' % kappa)
# # # ROC AUC
# # auc = roc_auc_score(yt, yp,multi_class='ovo')
# # print('ROC AUC: %f' % auc)
# # # confusion matrix
# matrix = confusion_matrix(yt, yp)
# print(matrix)