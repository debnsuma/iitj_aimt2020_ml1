# Author : Suman Debnath
# Email : debnath.1@iitj.ac.in
# Roll No : MT19AIE321
# M.Tech-AI(2020) 
# Date : 10th April 2020

#################################
#  Solution for Question No. 2  #
#################################

# Importing the modules 
import pandas as pd
import numpy as np
import operator

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

print_divider = "*"* 100

## Training with KNN

def hyper_combination_generator(no_trees):
    
    # learning rate 
    learning_rate = 1

    hyper_prams_combi = []
    for n in no_trees:
        _ = n, learning_rate
        hyper_prams_combi.append(_)
        
    return hyper_prams_combi

def all_clf(hyper_prams):
    all_clf_dict = {}
    for h_pram in hyper_prams:
        name = f"clfs_trees_{h_pram[0]}"
        n_estimators, learning_rate = h_pram[0], h_pram[1]
        all_clf_dict[name] = AdaBoostClassifier(n_estimators=n_estimators,
                                                learning_rate=learning_rate,
                                                random_state=42)
    
    return all_clf_dict

def clf_fit(clfs, X_train, y_train):
    for clf in clfs.values():
        clf.fit(X_train, y_train)
    
    return clfs

def clf_preditions(data, all_clf_dict):
    clf_pred = {}
    for c in all_clf_dict:
        clf_pred[c] = all_clf_dict[c].predict(data)
    
    return clf_pred

def clf_accuracy_score(y_true, clf_preds):
    clf_accuracy_score = {}
    for c in clf_preds:
        clf_accuracy_score[c] = accuracy_score(clf_preds[c], y_true)
    
    return clf_accuracy_score

def clf_error_score(clf_accuracy_score):
    
    clf_error_score = {clf: 1- clf_accuracy_score[clf] for clf in clf_accuracy_score.keys()}
    
    return clf_error_score

def clf_confusion_matrix(y_true, clf_preds):
    clf_cms = {}
    for c in clf_preds:
        clf_cms[c] = confusion_matrix(clf_preds[c], y_true)
    return clf_cms

def clf_f1_score(y_true, clf_preds):
    clf_f1s = {}
    for c in clf_preds:
        clf_f1s[c] = f1_score(clf_preds[c], y_true)
    return clf_f1s

## PART I
## Loading the dataset

df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
Y = df["target"]

## Splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.20, random_state=42)

## Training the model
# number of trees used in the classifier 
no_trees = range(20, 260, 20)

hyper_prams = hyper_combination_generator(no_trees)
all_clf_dict = all_clf(hyper_prams)
all_clf_dict = clf_fit(all_clf_dict, X_train, y_train)

# **(a) Plot the `TEST` error of classifier vs number of trees used in the classifier (varying n from range [20,40,60,80,...240] at an interval of 20).**

clf_preds_test = clf_preditions(X_test, all_clf_dict)
clf_accuracy_score_test = clf_accuracy_score(y_test, clf_preds_test)
clf_error_score_test = clf_error_score(clf_accuracy_score_test)

x = list(clf_error_score_test.keys())
y = list(clf_error_score_test.values())

plt.figure(figsize=(10,10))

plt.plot(x, y, label="Test Data", marker='o', markerfacecolor='red', markersize=10)

plt.xlabel('Different Classifiers')
plt.ylabel('Error(Testing)')
plt.title('Model Evaluations')
plt.grid(axis='x', color='0.95')
plt.legend(loc='upper center')
plt.xticks(rotation=90)
#plt.show()

plt.savefig("Model_Evaluations_Test.png")
print(print_divider)
print("Model Evaluations with TEST data is saved in: 'Model_Evaluations_Test.png'")
print("TEST error of classifier vs number of trees used in the classifier")
print(clf_error_score_test)
print(print_divider)

# **(b) Plot the `TRAIN` error of classifier vs no. of trees used in the classifier (varying n from range [20,40,60,80,...240] at an interval of 20).**

clf_preds_train = clf_preditions(X_train, all_clf_dict)
clf_accuracy_score_train = clf_accuracy_score(y_train, clf_preds_train)
clf_error_score_train = clf_error_score(clf_accuracy_score_train)

x = list(clf_error_score_train.keys())
y = list(clf_error_score_train.values())
plt.figure(figsize=(10,10))

plt.plot(x, y, label="Train Data", marker='o', markerfacecolor='red', markersize=10)

plt.xlabel('Different Classifiers')
plt.ylabel('Error(Training)')
plt.title('Model Evaluations')
plt.grid(axis='x', color='0.95')
plt.legend(loc='upper center')
plt.xticks(rotation=90)
# plt.show()

plt.savefig("Model_Evaluations_Train.png")
print("Model Evaluations with TRAIN data is saved in: 'Model_Evaluations_Train.png'")
print("TRAIN error of classifier vs number of trees used in the classifier")
print(clf_error_score_train)
print(print_divider)
# **(c) Create ensemble of classifiers with varying n in (50,100,150,200).**

# number of trees used in the classifier 
no_trees = [50,100,150,200]
hyper_prams = hyper_combination_generator(no_trees)
all_clf_dict = all_clf(hyper_prams)

collection_of_all_clf = list(all_clf_dict.items())
eclf1 = VotingClassifier(estimators=collection_of_all_clf, voting='hard')

eclf1 = eclf1.fit(X_train, y_train)
eclf1_preds = eclf1.predict(X_test)

print("Classifiers Performance when we used ensemble of classifiers with varying number of trees(50,100,150,200)")
print(classification_report(y_test, eclf1_preds))

cm = confusion_matrix(y_test, eclf1_preds)
f1 = f1_score(y_test, eclf1_preds)
accuracy = accuracy_score(y_test, eclf1_preds)

print(f"Accuracy : {accuracy}")
print(f"F1 Score : {f1}")
print(f"Confusion Matrix : \n{cm}")
print(print_divider)

# **(e) Compare the ensemble classifier obtained in (c) with single decision tree classifier trained on the training data (split criterion = entropy), report which is better and why.**

clf_tree = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=1)
clf_tree = clf_tree.fit(X_train, y_train)
dt_preds = clf_tree.predict(X_test)

print("Classifiers Performance comparision between A) ensemble of classifiers with varying number of trees(50,100,150,200) and B) Single decision tree classifier") 
print(classification_report(y_test, dt_preds))

cm = confusion_matrix(y_test, dt_preds)
f1 = f1_score(y_test, dt_preds)
accuracy = accuracy_score(y_test, dt_preds)

print(f"Accuracy : {accuracy}")
print(f"F1 Score : {f1}")
print(f"Confusion Matrix : \n{cm}")
print(print_divider)
print("Conclusion: The performance of the Single Decision Tree is better than the Ensemble Classifier we created with different no. of trees")
print(print_divider)

