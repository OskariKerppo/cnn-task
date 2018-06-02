#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import read_yale
from sklearn import svm
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
	#We leave 20 % of the data for final validation. This is not used in training
	final_validation = images.sample(frac=0.2)
	images = images.loc[~images.index.isin(final_validation.index)]
	final_validation_labels = final_validation.index.get_level_values('person').values
	final_validation_labels = convert_to_hot(final_validation_labels)
	final_pic_names = final_validation.index.get_level_values('pic_name').values
	final_validation = final_validation.values

	total_training = images
	total_labels = total_training.index.get_level_values('person').values
	total_labels = convert_to_hot(total_labels)
	total_training = total_training.values


	print("Loaded dataset and separated final validation data")
	print("Time passed: " + str(time.time()-start))
	#We use 10-fold cross-validation. The accuracy of the 10 SVM's is then calculated on majority vote on final 
	#validation data
	"""
	k_fold_data = {}
	k_fold_labels = {}
	k_f = k_fold
	print("Separating remaining data into " + str(k_fold) + " subsets...")
	#SPLIT TRAINING DATA TO 10 SUBSETS RANDOMLY
	for i in range(k_fold):
		print("Fraction to append: " + str(1.0/k_f))
		k_fold_data[i] = images.sample(frac=1.0/k_f)
		images = images.loc[~images.index.isin(k_fold_data[i].index)]
		k_fold_labels[i] = k_fold_data[i].index.get_level_values('person').values
		k_fold_data[i] = k_fold_data[i].values
		k_f -= 1


	print("Data ready.")
	print("Time passed: " + str(time.time()-start))
	#TRAIN 10 SVMs
	print("Training SVMs...")
	accuracies = []
	for i in range(k_fold):
		print("Training model: " + str(i+1)+ "...")
		formatted = False
		for key in k_fold_data:
			if key == i:
				validation_data = k_fold_data[key]
				validation_labels = k_fold_labels[key]
			elif not formatted:
				training_data = k_fold_data[key]
				training_labels = k_fold_labels[key]
				formatted = True
			else:
				training_data = np.vstack([training_data, k_fold_data[key]])
				training_labels = np.append(training_labels, k_fold_labels[key])
		print("Validation data and label shape: ")
		print(validation_data.shape)
		print(validation_labels.shape)
		print("Training data and label shape: ")
		print(training_data.shape)
		print(training_labels.shape)
		clf = svm.LinearSVC()
		#print(training_data)
		#print(training_labels)
		clf.fit(training_data,training_labels)
		acc_pred = clf.predict(validation_data)
		acc = accuracy_score(validation_labels,acc_pred)
		print("Accuracy: " + str(acc))
		#print(validation_labels)
		#print(validation_labels.shape)
		#print(acc_pred)
		#print(acc_pred.shape)
		accuracies.append(acc)
		with open(model_folder + r'\svm_'+str(i)+'.pickle','wb') as file:
			pickle.dump(clf,file)
		print("Model "+ str(i+1)+" trained!")
		print("Time passed: " + str(time.time()-start))

	with open(model_folder+r'\accuracies.pickle','wb') as f:
		pickle.dump(accuracies,f)
	print("Cross validation ready. Training final model...")
	clf = svm.LinearSVC()
	clf.fit(total_training,total_labels)
	with open(model_folder + r'\svm_total.pickle','wb') as file:
		pickle.dump(clf,file)

	print("All models trained!")

	with bz2.BZ2File(model_folder + r'\test_set.pbz2','w') as file:
		pickle.dump(final_validation,file)
	with bz2.BZ2File(model_folder + r'\test_labels.pbz2','wb') as file:
		pickle.dump(final_validation_labels,file)
	with bz2.BZ2File(model_folder + r'\test_pic_names.pbz2','wb') as file:
		pickle.dump(final_pic_names,file)
	"""


	
if __name__ == "__main__":
	main()


