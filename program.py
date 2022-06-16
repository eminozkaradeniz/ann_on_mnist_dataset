# -*- coding: utf-8 -*-


# Importing the libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD


# Part 1 - Preprocessing
# Importing images and flattening
labels = os.listdir('mnist-train')
targets = []
images = []

for label in labels:
    imgs = os.listdir(f'mnist-train/{label}')
    for img in imgs:
        images.append(np.array(Image.open(f'mnist-train/{label}/{img}')).flatten())
        targets.append(label)
        
images = np.array(images)
df_train = pd.DataFrame(images)
df_train['target'] = pd.Series(targets)     
        
labels = os.listdir('mnist-test')
targets = []
images = []

for label in labels:
    imgs = os.listdir(f'mnist-test/{label}')
    for img in imgs:
        images.append(np.array(Image.open(f'mnist-test/{label}/{img}')).flatten())
        targets.append(label)
        
images = np.array(images)
df_test = pd.DataFrame(images)
df_test['target']=pd.Series(targets)   


# Shuffling the dataset
from sklearn.utils import shuffle
df_train = shuffle(df_train)
df_train.reset_index(inplace=True, drop=True)
df_test = shuffle(df_test)
df_test.reset_index(inplace=True, drop=True)


# Splitting the dataset into Training set and Test set
X_train, y_train = np.array(df_train.drop('target', axis=1)), df_train['target']
X_test, y_test = np.array(df_test.drop('target', axis=1)), df_test['target']


# One Hot Encoding the dependent variable for multiclass classification
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, 10)


# Feature Scaling
X_train = X_train/255
X_test = X_test/255


# Part 2 - Building the ANN
# Initializing the ANN
model = tf.keras.Sequential()

# Input layer
model.add(layers.InputLayer(input_shape=(784,)))

# First hidden layer
model.add(layers.Dense(392, activation='relu'))

# Second hidden layer
model.add(layers.Dense(196, activation='relu'))

# Output layer
model.add(layers.Dense(10, activation='softmax'))


initial_learning_rate = 0.001
decay_rate = 0.01


# Learning rate scheduler
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, 100000, decay_rate, staircase=True)
    

# SGD Optimizer
optimizer = SGD(learning_rate=lr_scheduler, 
                momentum=0.99,
                nesterov=True)


# Compiling the ANN
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,
              metrics=['accuracy'])


# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=25)


# Part 3 - Training the ANN
model.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[early_stop],
          validation_data=(X_test, y_test))


# Part 4 - Evaluating the Model
y_test = np.argmax(y_test, axis = -1)
y_train = np.argmax(y_train, axis= -1)

y_pred = np.argmax(model.predict(X_test), axis=-1)
y_pred_train = np.argmax(model.predict(X_train), axis=-1)

from sklearn.metrics import classification_report
print('\nTrain')
print(classification_report(y_train, y_pred_train))
print('\nTest')
print(classification_report(y_test, y_pred))
