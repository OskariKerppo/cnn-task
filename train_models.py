#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import read_yale
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

k_fold = 3 # CHANGE TO 10 IN FINAL VERSION!!!


def hot_label(yaleB0x):
	yale_index = int(yaleB0x[5:])
	yale_array = np.array([])
	for i in range(39):
		if yale_index -1 == i:
			yale_array = np.append(yale_array,1)
		else:
			yale_array = np.append(yale_array,0)
	return yale_array

def convert_to_hot(array):
	hot_array = np.array([])
	for i in range(len(array)):
		if i == 0:
			hot_array = np.append(hot_array,hot_label(array[i]))
		else:
			hot_array = np.vstack([hot_array,hot_label(array[i])])
	return hot_array


def main():
	print("Initializing...")
	code_folder = os.getcwd()
	model_folder = code_folder + r'\Trainded_Models'
	files = [f for f in listdir(model_folder) if isfile(join(model_folder, f))]
	for file in files:
		try:
			os.remove(model_folder+'\\'+file)
		except:
			print("File deletion failed! " + file)
	start = time.time()
	images, resolution = read_yale.get_croppedyale_as_df()
	images = shuffle(images)
	#We leave 20 % of the data for final testing. This is not used in training
	test_X = images.sample(frac=0.2)
	images = images.loc[~images.index.isin(test_X.index)]
	test_labels = test_X.index.get_level_values('person').values
	test_labels = convert_to_hot(test_labels)
	test_pic_names = test_X.index.get_level_values('pic_name').values
	test_X = test_X.values
	test_X = np.reshape(test_X,(-1,32256,1))

	train_X = images.sample(frac=0.7)
	images = images.loc[~images.index.isin(train_X.index)]
	train_label = train_X.index.get_level_values('person').values
	train_label = convert_to_hot(train_label)
	train_X = train_X.values
	train_X = np.reshape(train_X,(-1,32256,1))

	valid_X = images
	valid_label = valid_X.index.get_level_values('person').values
	valid_label = convert_to_hot(valid_label)
	valid_X = valid_X.values
	valid_X = np.reshape(valid_X,(-1,32256,1))


	print("Loaded dataset and separated final validation data")
	print("Time passed: " + str(time.time()-start))

	#Training convolutional NN with keras
	print("Defining model")
	print("Time passed: " + str(time.time()-start))
	batch_size = 64
	epochs = 40
	num_classes = 39

	cudnn = Sequential()
	cudnn.add(Conv1D(16, kernel_size=5,activation='relu',input_shape=(32256,1),padding='same'))
	cudnn.add(MaxPooling1D(pool_size=2,padding='same'))
	cudnn.add(Conv1D(36, kernel_size=5,activation='relu',input_shape=(32256,1),padding='same'))
	cudnn.add(MaxPooling1D(pool_size=2,padding='same'))
	cudnn.add(Flatten())
	cudnn.add(Dense(512, activation='relu'))  
	cudnn.add(Dropout(rate=0.5))	               
	cudnn.add(Dense(num_classes, activation='softmax'))
	print("Time passed: " + str(time.time()-start))
	print("Compiling model")
	#Compile model
	cudnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
	print("Time passed: " + str(time.time()-start))
	cudnn.summary()
	#Train model for 20 epochs
	print("Training model")
	start_training = time.time()
	cudnn_train = cudnn.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
	print("Total training time: " + str(time.time()-start_training))
	print("Time passed: " + str(time.time()-start))
	#Save model for later use
	cudnn.save(model_folder + r'\cudnn.h5')


	#Dump test sets


	accuracy = cudnn_train.history['acc']
	val_accuracy = cudnn_train.history['val_acc']
	loss = cudnn_train.history['loss']
	val_loss = cudnn_train.history['val_loss']

	
	with bz2.BZ2File(model_folder + r'\accuracy.pbz2','w') as file:
		pickle.dump(accuracy,file)
	with bz2.BZ2File(model_folder + r'\val_accuracy.pbz2','w') as file:
		pickle.dump(val_accuracy,file)		
	with bz2.BZ2File(model_folder + r'\loss.pbz2','w') as file:
		pickle.dump(loss,file)
	with bz2.BZ2File(model_folder + r'\val_loss.pbz2','w') as file:
		pickle.dump(val_loss,file)

	with bz2.BZ2File(model_folder + r'\test_set.pbz2','w') as file:
		pickle.dump(test_X,file)
	with bz2.BZ2File(model_folder + r'\test_labels.pbz2','wb') as file:
		pickle.dump(test_labels,file)
	with bz2.BZ2File(model_folder + r'\test_pic_names.pbz2','wb') as file:
		pickle.dump(test_pic_names,file)






if __name__ == "__main__":
	main()


