# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:43:45 2023

@author: casey
"""

# ---------------------------------------------------------------------------------------------------------------------------- #
## IMPORTS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow.keras
#from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import nltk
import string
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re

# ---------------------------------------------------------------------------------------------------------------------------- #
## LOAD DATA

topics_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Fall_2023/Neural_Nets/Final_Exam/data/Final_News_DF_Labeled_ExamDataset.csv')

# --------------------------------------------------------------------------------------- #
## CONVERT CATEGORICAL LABELS TO NUMERIC LABELS

# 0:Football, 1:Politics, 2:Science
le = LabelEncoder()
topics_df['LABEL'] = le.fit_transform(topics_df['LABEL'])

texts = topics_df.iloc[:,1:]
labels = topics_df['LABEL']

# --------------------------------------------------------------------------------------- #
## CREATE TRAIN TEST VAL DATA
train_text, test_text, train_labels, test_labels = train_test_split(texts, labels, test_size=150, random_state=1)

## Create validation data
train_text, val_text, train_labels, val_labels = train_test_split(train_text, train_labels, test_size=150, random_state=1)


# ---------------------------------------------------------------------------------------------------------------------------- #
## VERIFY SHAPES

## Set the input shape
train_input_shape=train_text.shape
test_input_shape=test_text.shape
val_input_shape=val_text.shape

# check shapes
print("The input shape for the training reviews is\n", train_input_shape) ## (1193, 300)
print("The input shape for the testing reviews is\n", test_input_shape) ## (150. 300)
print("The input shape for the validation reviews is\n",val_input_shape) ## (150, 300)

print('The shape of train_labels is:', train_labels.shape) # should be of shape (1193,)
print('The shape of test_labels is:', test_labels.shape) # should be of shape (150,)
print('The shape of val_labels is:', val_labels.shape) # should be of shape (150,)

# ---------------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------------- #
## ANN MODEL
# ---------------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------------------------------------------------- #
## BUILD THE INITIAL ANN MODEL

ANN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='sigmoid'), 
  #tf.keras.layers.Dropout(0.2), 
  #tf.keras.layers.Dense(10, activation='sigmoid'),
  #tf.keras.layers.Dropout(0.2), 
  tf.keras.layers.Dense(3, activation='softmax'), 
])

ANN_Model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer='adam')

# don't use early stopping
Hist=ANN_Model.fit(train_text, train_labels, epochs=100, validation_data=(val_text, val_labels))

ANN_Model.summary()

# ---------------------------------------------------------------------------------------------------------------------------- #
## BUILD THE FINAL ANN MODEL

# create a callback for early stopping to prevent overfitting
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, mode='min')

ANN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='sigmoid'), 
  tf.keras.layers.Dropout(0.2), 
  tf.keras.layers.Dense(10, activation='sigmoid'), 
  #tf.keras.layers.Dense(10, activation='sigmoid'),
  #tf.keras.layers.Dropout(0.2), 
  tf.keras.layers.Dense(3, activation='softmax'), 
])

ANN_Model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer='adam')


# use early stopping
Hist=ANN_Model.fit(train_text, train_labels, epochs=100, validation_data=(val_text, val_labels), callbacks=[callback])

ANN_Model.summary()

# ---------------------------------------------------------------------------------------------------------------------------- #
## ACCURACY AND LOSS PLOTS

# train accuracy and val accuracy 
plt.plot(Hist.history['accuracy'], label='train_accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')

# train loss and val loss
plt.plot(Hist.history['loss'], label='train_loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1.5])
plt.legend(loc='lower right')

# ---------------------------------------------------------------------------------------------------------------------------- #
## ACCURACY ON TEST SET
test_loss, test_acc = ANN_Model.evaluate(test_text,  test_labels, verbose=2)

print(test_acc)

# ---------------------------------------------------------------------------------------------------------------------------- #
## GET MODEL PREDICTIONS ON TEST SET
ANNpredictions=ANN_Model.predict([test_text])
print(ANNpredictions)
print(ANNpredictions.shape)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
print("The prediction accuracy via confusion matrix is:\n")
ANN_Max_Values = np.squeeze(np.array(ANNpredictions.argmax(axis=1)))
print(ANN_Max_Values)
print(np.argmax([ANNpredictions]))
print(confusion_matrix(y_pred=ANN_Max_Values, y_true=test_labels))

# ---------------------------------------------------------------------------------------------------------------------------- #
## PRETTY CONFUSION MATRIX
labels = [0, 1, 2]
cm = confusion_matrix(y_true=ANN_Max_Values, y_pred=test_labels, labels=labels)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt     

fig, ax = plt.subplots(figsize=(13,13)) 
#ax= plt.subplot()
#sns.set(font_scale=3)
#sns.set (rc = {'figure.figsize':(40, 40)})
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
#annot=True to annotate cells, ftm='g' to disable scientific notation
# annot_kws si size  of font in heatmap
# labels, title and ticks
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: ANN') 
ax.xaxis.set_ticklabels(["0:Football","1:Politics","2:Science"],rotation=90, fontsize = 18)

ax.yaxis.set_ticklabels(["0:Football","1:Politics","2:Science"],rotation=0, fontsize = 18)

# ---------------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------------- #
## CNN MODEL
# ---------------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------------------------------------------------- #
## BUILD THE INITIAL CNN MODEL

CNN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=train_input_shape[1], output_dim=32, input_length=train_input_shape[1]),
  tf.keras.layers.Conv1D(filters=50, kernel_size=3, activation='relu'),
  tf.keras.layers.MaxPool1D(pool_size=2),
  
  tf.keras.layers.Flatten(),
 
  tf.keras.layers.Dense(20),
  tf.keras.layers.Dropout(0.5),
 
  tf.keras.layers.Dense(3, activation="softmax")
])

CNN_Model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer='adam')

# don't use early stopping
Hist=CNN_Model.fit(train_text, train_labels, epochs=100, validation_data=(val_text, val_labels))

CNN_Model.summary()

# ---------------------------------------------------------------------------------------------------------------------------- #
## BUILD THE FINAL CNN MODEL

# create a callback for early stopping to prevent overfitting
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

CNN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=train_input_shape[1], output_dim=8, input_length=train_input_shape[1]),
  tf.keras.layers.Conv1D(10, kernel_size=3, activation='relu'),
  tf.keras.layers.MaxPool1D(pool_size=2),
  
  tf.keras.layers.Flatten(),
 
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  
  tf.keras.layers.Dense(3, activation='softmax')
])

CNN_Model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer='adam')


# use early stopping
Hist=CNN_Model.fit(train_text, train_labels, epochs=100, validation_data=(val_text, val_labels), callbacks=[callback])

CNN_Model.summary()

# ---------------------------------------------------------------------------------------------------------------------------- #
## ACCURACY AND LOSS PLOTS

# train accuracy and val accuracy 
plt.plot(Hist.history['accuracy'], label='train_accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')

# train loss and val loss
plt.plot(Hist.history['loss'], label='train_loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1.2])
plt.legend(loc='lower right')

# ---------------------------------------------------------------------------------------------------------------------------- #
## ACCURACY ON TEST SET
test_loss, test_acc = CNN_Model.evaluate(test_text,  test_labels, verbose=2)

print(test_acc)

# ---------------------------------------------------------------------------------------------------------------------------- #
## GET MODEL PREDICTIONS ON TEST SET
CNNpredictions=CNN_Model.predict([test_text])
print(CNNpredictions)
print(CNNpredictions.shape)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
print("The prediction accuracy via confusion matrix is:\n")
CNN_Max_Values = np.squeeze(np.array(CNNpredictions.argmax(axis=1)))
print(CNN_Max_Values)
print(np.argmax([CNNpredictions]))
print(confusion_matrix(y_pred=CNN_Max_Values, y_true=test_labels))

# ---------------------------------------------------------------------------------------------------------------------------- #
## PRETTY CONFUSION MATRIX
labels = [0, 1, 2]
cm = confusion_matrix(y_true=CNN_Max_Values, y_pred=test_labels, labels=labels)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt     

fig, ax = plt.subplots(figsize=(13,13)) 
#ax= plt.subplot()
#sns.set(font_scale=3)
#sns.set (rc = {'figure.figsize':(40, 40)})
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
#annot=True to annotate cells, ftm='g' to disable scientific notation
# annot_kws si size  of font in heatmap
# labels, title and ticks
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: CNN') 
ax.xaxis.set_ticklabels(["0:Football","1:Politics","2:Science"],rotation=90, fontsize = 18)

ax.yaxis.set_ticklabels(["0:Football","1:Politics","2:Science"],rotation=0, fontsize = 18)

# ---------------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------------- #
## LSTM MODEL
# ---------------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------------- #
## BUILD THE INITIAL LSTM MODEL

LSTM_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=train_input_shape[1], output_dim=32, input_length=train_input_shape[1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
  tf.keras.layers.Dense(3, activation="softmax")
])

LSTM_Model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer='adam')

# don't use early stopping
Hist=LSTM_Model.fit(train_text, train_labels, epochs=100, validation_data=(val_text, val_labels))

LSTM_Model.summary()

# ---------------------------------------------------------------------------------------------------------------------------- #
## BUILD THE FINAL LSTM MODEL

LSTM_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=train_input_shape[1], output_dim=32, input_length=train_input_shape[1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(3, activation="softmax")
])

LSTM_Model.compile(loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  optimizer='adam')


# use early stopping
Hist=LSTM_Model.fit(train_text, train_labels, epochs=40, validation_data=(val_text, val_labels))

LSTM_Model.summary()

# ---------------------------------------------------------------------------------------------------------------------------- #
## ACCURACY AND LOSS PLOTS

# train accuracy and val accuracy 
plt.plot(Hist.history['accuracy'], label='train_accuracy')
plt.plot(Hist.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')

# train loss and val loss
plt.plot(Hist.history['loss'], label='train_loss')
plt.plot(Hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1.4])
plt.legend(loc='lower right')

# ---------------------------------------------------------------------------------------------------------------------------- #
## ACCURACY ON TEST SET
test_loss, test_acc = LSTM_Model.evaluate(test_text,  test_labels, verbose=2)

print(test_acc)

# ---------------------------------------------------------------------------------------------------------------------------- #
## GET MODEL PREDICTIONS ON TEST SET
LSTMpredictions=LSTM_Model.predict([test_text])
print(LSTMpredictions)
print(LSTMpredictions.shape)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
print("The prediction accuracy via confusion matrix is:\n")
LSTM_Max_Values = np.squeeze(np.array(LSTMpredictions.argmax(axis=1)))
print(LSTM_Max_Values)
print(np.argmax([LSTMpredictions]))
print(confusion_matrix(y_pred=LSTM_Max_Values, y_true=test_labels))

# ---------------------------------------------------------------------------------------------------------------------------- #
## PRETTY CONFUSION MATRIX  
labels = [0, 1, 2]
cm = confusion_matrix(y_true=LSTM_Max_Values, y_pred=test_labels, labels=labels)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt     

fig, ax = plt.subplots(figsize=(13,13)) 
#ax= plt.subplot()
#sns.set(font_scale=3)
#sns.set (rc = {'figure.figsize':(40, 40)})
sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
#annot=True to annotate cells, ftm='g' to disable scientific notation
# annot_kws si size  of font in heatmap
# labels, title and ticks
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: LSTM') 
ax.xaxis.set_ticklabels(["0:Football","1:Politics","2:Science"],rotation=90, fontsize = 18)

ax.yaxis.set_ticklabels(["0:Football","1:Politics","2:Science"],rotation=0, fontsize = 18)