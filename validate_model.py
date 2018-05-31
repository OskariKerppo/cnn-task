from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import time
import os
from os import listdir
from os.path import isfile, join
from train_models import k_fold
import itertools
import matplotlib.pyplot as plt
import bz2



start = time.time()


code_folder = os.getcwd()
model_folder = code_folder + r'\Trainded_Models'


files = [f for f in listdir(model_folder) if isfile(join(model_folder, f))]
print(files)
pickled_svm = [s for s in files if 'svm' in s and 'total' not in s]
if not pickled_svm:
     train_models.main()
pickled_svm = [s for s in files if 'svm' in s and 'total' not in s]
total_svm = [s for s in files if 'svm' in s and 'total' in s]
total_svm = total_svm[0] 
SVMs = {}
for i, svm in enumerate(pickled_svm):
     SVMs[i] = pickle.load(open(model_folder + '\\' + svm, 'rb'))

with bz2.BZ2File(model_folder+r'\test_set.pbz2','r') as f:
        final_validation = pickle.load(f)
with bz2.BZ2File(model_folder+r'\test_labels.pbz2','r') as f:
        final_validation_labels = pickle.load(f)
with bz2.BZ2File(model_folder+r'\test_pic_names.pbz2','r') as f:
        final_pic_names = pickle.load(f)

SVMs['total'] = pickle.load(open(model_folder + '\\' + total_svm, 'rb'))

accuracies = pickle.load(open(model_folder+r'\accuracies.pickle','rb'))
print(accuracies)
scores = np.array(accuracies)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



print("Final model validation in process...")
predictions = np.array([],dtype=str)
confidences = np.array([],dtype=float)
print("Validatin models")
for i in range(k_fold):
        print("Validatin model " + str(i+1) + "...")
        clf = SVMs[i]
        if i == 0:
                predictions = clf.predict(final_validation)
                confidence = clf.decision_function(final_validation)
                cp = []
                for j in range(len(confidence)):
                    cp.append(max(confidence[j]))
                confidences = np.append(confidences,cp) 
                print("Model accuracy: " + str(accuracy_score(final_validation_labels,predictions)))
        else:
                prediction = clf.predict(final_validation)
                predictions = np.vstack([predictions, prediction])
                cp = []
                for j in range(len(confidence)):
                    cp.append(max(confidence[j]))
                confidences = np.vstack([confidences,cp])
                print("Model accuracy: " + str(accuracy_score(final_validation_labels,prediction)))
        print("Validated model "+ str(i+1)+"!")
        print("Time passed: " + str(time.time()-start))

print("Final model predictions...")
clf = SVMs['total']
total_predictions = clf.predict(final_validation)
print("Final model accuracy: " + str(accuracy_score(final_validation_labels,total_predictions)))

print("All predictions ready! Combining results...")
final_predictions_vote = np.array([],dtype=str)
final_predictions_confidence = np.array([],dtype=str)


for i in range(len(predictions[0])):
	p_col = predictions[:,[i]]
	p_col = p_col.flatten()
	mode = Counter(p_col).most_common(1)[0][0]
	final_predictions_vote = np.append(final_predictions_vote,mode)

for i in range(len(predictions[0])):
     c_col = confidences[:,[i]]
     max_conf = np.argmax(c_col)
     highest_conf = predictions[max_conf][i]
     final_predictions_confidence = np.append(final_predictions_confidence,highest_conf)
print("Final predictions ready!")
print("Time passed: " + str(time.time()-start))


print("Final predictions")
#print(final_predictions)
print("Validation labels")
#print(final_validation_labels)
print("Majority vote accuracy")
correct = 0
incorrect = 0
incorrect_pics = []
print("Elements in list: correct label, pic name, predicted label")
for i in range(len(final_predictions_vote)):
	if final_predictions_vote[i] == final_validation_labels[i]:
		correct += 1
	else:
		incorrect += 1
		incorrect_pics.append([final_validation_labels[i],final_pic_names[i],final_predictions_vote[i]])

print("Correctly classified: " + str(correct))
print("Total validation samples: " + str(len(final_validation_labels)))
print("Percentage correct: " + str(correct/(correct + incorrect)))
print("Incorrectly classified pictures: ")
print(str(incorrect_pics))

print("Max confidence accuracy")
correct = 0
incorrect = 0
incorrect_pics = []
print("Elements in list: correct label, pic name, predicted label")
for i in range(len(final_predictions_confidence)):
     if final_predictions_confidence[i] == final_validation_labels[i]:
          correct += 1
     else:
          incorrect += 1
          incorrect_pics.append([final_validation_labels[i],final_pic_names[i],final_predictions_confidence[i]])

print("Correctly classified: " + str(correct))
print("Total validation samples: " + str(len(final_validation_labels)))
print("Percentage correct: " + str(correct/(correct + incorrect)))
print("Incorrectly classified pictures: ")
print(str(incorrect_pics))

print("Final model accuracy")
correct = 0
incorrect = 0
incorrect_pics = []
print("Elements in list: correct label, pic name, predicted label")
for i in range(len(total_predictions)):
     if total_predictions[i] == final_validation_labels[i]:
          correct += 1
     else:
          incorrect += 1
          incorrect_pics.append([final_validation_labels[i],final_pic_names[i],total_predictions[i]])

print("Correctly classified: " + str(correct))
print("Total validation samples: " + str(len(final_validation_labels)))
print("Percentage correct: " + str(correct/(correct + incorrect)))
print("Incorrectly classified pictures: ")
print(str(incorrect_pics))


print("Finally the majority vote of these 3 methods...")
combined_predictions = total_predictions
combined_predictions = np.vstack([combined_predictions,final_predictions_confidence])
combined_predictions = np.vstack([combined_predictions,final_predictions_vote])
combined_predictions_voted = np.array([],dtype=str)
for i in range(len(combined_predictions[0])):
     p_col = combined_predictions[:,[i]]
     p_col = p_col.flatten()
     mode = Counter(p_col).most_common(1)[0][0]
     combined_predictions_voted = np.append(combined_predictions_voted,mode)

print("Combined accuracy")
correct = 0
incorrect = 0
incorrect_pics = []
print("Elements in list: correct label, pic name, predicted label")
for i in range(len(combined_predictions_voted)):
     if combined_predictions_voted[i] == final_validation_labels[i]:
          correct += 1
     else:
          incorrect += 1
          incorrect_pics.append([final_validation_labels[i],final_pic_names[i],combined_predictions_voted[i]])

print("Correctly classified: " + str(correct))
print("Total validation samples: " + str(len(final_validation_labels)))
print("Percentage correct: " + str(correct/(correct + incorrect)))
print("Incorrectly classified pictures: ")
print(str(incorrect_pics))



class_names = []
for label in final_validation_labels:
	if int(label[5:]) not in class_names:
		class_names.append(int(label[5:]))
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
cnf_matrix = confusion_matrix(final_validation_labels, total_predictions)
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


#print(conf_matrix)
print("All tasks completed succesfully!")
print("Total time passed: " + str(time.time()-start))


