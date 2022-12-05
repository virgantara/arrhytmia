import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

le = LabelEncoder()

scaler = MinMaxScaler()
dataset = pd.read_csv('mfcc_features_gender.csv')

dataX = dataset.iloc[:,0:13]
y = dataset.iloc[:,13]
y = le.fit_transform(y)

outputs = tf.keras.utils.to_categorical(y=y, num_classes=2)
X = scaler.fit_transform(dataX)
X_train, X_test, y_train, y_test = train_test_split(X, outputs, test_size=0.3, random_state=42)


model = tf.keras.Sequential([
    layers.Dense(30,input_dim=X_train.shape[1],activation='relu'),
    # layers.Dense(30,activation='relu'),
    layers.Dense(30,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(X, y, validation_split=0.33, epochs=30,batch_size=10)
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # plt.show()
# plt.savefig('Accuracy.png')
#
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # plt.show()
# plt.savefig('Loss.png')