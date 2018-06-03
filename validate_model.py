from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import cPickle as pickle
from collections import Counter
import time
import os
import bz2
from os import listdir
from os.path import isfile, join

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

import matplotlib.pyplot as plt

code_folder = os.getcwd()
model_folder = code_folder + r'\Trainded_Models'

#Load dump

cudnn = load_model(model_folder + r'\cudnn.h5')


with bz2.BZ2File(model_folder+r'\accuracy.pbz2','r') as f:
        accuracy = pickle.load(f)

with bz2.BZ2File(model_folder+r'\val_accuracy.pbz2','r') as f:
        val_accuracy = pickle.load(f)

with bz2.BZ2File(model_folder+r'\loss.pbz2','r') as f:
        loss = pickle.load(f)

with bz2.BZ2File(model_folder+r'\val_loss.pbz2','r') as f:
        val_loss = pickle.load(f)

with bz2.BZ2File(model_folder+r'\test_set.pbz2','r') as f:
        test_X = pickle.load(f)

with bz2.BZ2File(model_folder+r'\test_set.pbz2','r') as f:
        test_X = pickle.load(f)

with bz2.BZ2File(model_folder+r'\test_labels.pbz2','r') as f:
        test_labels = pickle.load(f)

with bz2.BZ2File(model_folder+r'\test_pic_names.pbz2','r') as f:
        test_pic_names = pickle.load(f)



#Validate model
print("Validating model")
cudnn_eval = cudnn.evaluate(test_X, test_labels, verbose=0)
print('Test loss:', cudnn_eval[0])
print('Test accuracy:', cudnn_eval[1])




epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


print(np.array([test_X[0]]).shape)


#Predict some images
p1 = cudnn.predict(np.array([test_X[0]]))
p2 = cudnn.predict(np.array([test_X[0]]))
p3 = cudnn.predict(np.array([test_X[0]]))
print(p1,p2,p3)
