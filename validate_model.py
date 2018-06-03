from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
import itertools
import imageio
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

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

cudnn_prob = cudnn.predict(test_X)
cudnn_test_labels = cudnn_prob.argmax(axis=-1)


epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



#Confusion matrix

test_label_names = test_labels.argmax(axis=-1)
class_names = []
for label in test_label_names:
    if int(label) not in class_names:
        class_names.append(int(label))
class_names = sorted(class_names)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label_names, cudnn_test_labels)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



print(np.array([test_X[0]]).shape)


#Predict some images
p1 = cudnn.predict(np.array([test_X[0]]))
p2 = cudnn.predict(np.array([test_X[1]]))
p3 = cudnn.predict(np.array([test_X[2]]))
print(p1,p2,p3)



correct = 0
incorrect = 0
incorrect_pics = []
print("Elements in list: correct label, pic name, predicted label, predictred class probability, max predicted class probability")
for i in range(len(test_label_names)):
    if test_label_names[i] == cudnn_test_labels[i]:
        correct += 1
    else:
        incorrect += 1
        incorrect_pics.append([test_label_names[i],test_pic_names[i],cudnn_test_labels[i],cudnn_prob[i],cudnn_prob[i][cudnn_test_labels[i]]])

print("Correctly classified: " + str(correct))
print("Total validation samples: " + str(len(test_label_names)))
print("Percentage correct: " + str(float(correct)/(correct + incorrect)))
print("Incorrectly classified pictures: ")
print(str(incorrect_pics))


#Plot incorrectly classified images



pic_folder = code_folder + r'\CroppedYale'
f, axarr = plt.subplots(2,len(incorrect_pics))
for i in range(len(incorrect_pics)):
    true_person = incorrect_pics[i][0] + 1
    if len(str(true_person)) == 1:
        true_person = 'yaleB0' + str(true_person)
    else:
        true_person = 'yaleB' + str(true_person)

    pic_path = pic_folder + '\\' + true_person + '\\' + true_person + '_' + incorrect_pics[i][1] + '.pgm'
    pic = mpimg.imread(pic_path)

    #pic_gray = np.array(pic / 255.0)

    axarr[0,i].imshow(pic, cmap = cm.Greys_r)
    axarr[0,i].set_title("Incorrectly classified : {}".format(true_person))
    axarr[0,i].axis('off')

    predicted_person = incorrect_pics[i][2] + 1
    if len(str(predicted_person)) == 1:
        predicted_person = 'yaleB0' + str(predicted_person)
    else:
        predicted_person = 'yaleB' + str(predicted_person)

    pic_path = pic_folder + '\\' + predicted_person + '\\' + predicted_person + '_' + incorrect_pics[i][1] + '.pgm'
    pic = mpimg.imread(pic_path)

    pic_gray = np.array(pic / 255.0)

    axarr[1,i].imshow(pic, cmap = cm.Greys_r)
    axarr[1,i].set_title("Predicted person : {}".format(predicted_person))
    axarr[1,i].axis('off')


plt.show()
