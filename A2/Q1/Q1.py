# Author : Suman Debnath
# Email : debnath.1@iitj.ac.in
# Roll No : MT19AIE321
# M.Tech-AI(2020) 
# Date : 10th April 2020

#################################
#  Solution for Question No. 1  #
#################################

# Importing the modules 

import pandas as pd
import numpy as np
import operator

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
print_divider = "*"* 100

## Loading the dataset

data = load_wine()
data_array = data['data']
target_array = data['target']
data_column_name = data['feature_names']
data_target_name = data['target_names']

X = pd.DataFrame(data_array, columns=data_column_name)
Y = pd.Series(target_array)

## Spliting the data into training, test and validation

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=42, stratify=Y)

## PART I
print(print_divider)
#print("\n")
print("PART 1 of the Question")
#print("\n")

# **Plot the class-wise distribution of data in the train and test set (one histogram for train set, one for test set)**

## Histogram with TRAINING data
y_train_np = y_train.to_numpy()
labels, counts = np.unique(y_train_np, return_counts=True)
plt.figure(figsize=(8,8))
plot_train = plt.bar(labels, counts, align='center', color="orange")
plot_train = plt.gca().set_xticks(labels)
plot_train = plt.xlabel("Different Class")
plot_train = plt.ylabel("Frequency")
plot_train = plt.title("TRAINING Data Histogram")

plt.savefig("Histogram_with_training_data.png")
print("Histogram with TRAINING data is saved in: 'Histogram_with_training_data.png'")

# plt.show()

## Histogram with TEST data
y_test_np = y_test.to_numpy()
labels, counts = np.unique(y_test_np, return_counts=True)
plt.figure(figsize=(8,8))
plot_test = plt.bar(labels, counts, align='center')
plot_test = plt.gca().set_xticks(labels)
plot_test = plt.xlabel("Different Class")
plot_test = plt.ylabel("Frequency")
plot_test = plt.title("TESTING Data Histogram")

plt.savefig("Histogram_with_test_data.png")
print("Histogram with TEST data is saved in: 'Histogram_with_test_data.png'")
#plt.show()

labels, counts_test = np.unique(y_test_np, return_counts=True)
_, counts_train = np.unique(y_train_np, return_counts=True)

x = np.arange(len(labels))  # the label locations
width = 0.30  # the width of the bars

fig, ax = plt.subplots(figsize=(10,10))
rects1 = ax.bar(x - width/2, counts_test, width, label='TEST')
rects2 = ax.bar(x + width/2, counts_train, width, label='TRAINING')

## Histogram with TRAINING and TEST data

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Frequency')
ax.set_xticklabels('Different Class')
ax.set_title('TESTING vs TRAINING Data Histogram')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.xlabel("Different Class")
plt.ylabel("Frequency")
plt.title("TESTING vs TRAINING Data Histogram")

plt.savefig("Histogram_with_both_test_and_training_data.png")
print("Histogram with TEST and TRAINING data is saved in: 'Histogram_with_both_test_and_training_data.png'")

#plt.show()


from scipy import stats
st = stats.ks_2samp(y_train, y_test)

print(print_divider)
print(f"Kolmogorov-Smirnov statistic on these above two data set with TESTING and TRAIN Data set are:")
print(f"KS statistic: {st.statistic}")
print(f"Two-tailed p-value: {st.pvalue}")
print("As we see both statistics and pvalue value is LOW, we can conclude that dataset is coming from the same distribution.")


## PART II 
print(print_divider)
#print("\n")
print("PART 2(a) of the Question")
#print("\n")
print(print_divider)

## Train a Gaussian Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

for i in zip(clf.classes_, clf.class_prior_):
    print(f"Class Priority of 'Class Label {i[0]}' is {i[1]} ")
print(print_divider)

# The `mean` and `variance` of each feature per class

class_mean_df = pd.DataFrame(data=clf.theta_, columns=data_column_name, index=clf.classes_)
class_var_df = pd.DataFrame(data=clf.sigma_, columns=data_column_name, index=clf.classes_)

print("MEAN for each classes")
print(print_divider)
print(class_mean_df)

print(print_divider)
print("VARIANCE for each classes")
print(print_divider)
print(class_var_df)

class_mean_df.to_excel("mean_sheet.xls")
class_var_df.to_excel("variance_sheet.xls")

print(print_divider)
print("The MEAN of each feature per class is saved in the file 'mean_sheet.xls'")
print("The VARIANCE of each feature per class is saved in the file 'variance_sheet.xls'")
print(print_divider)

## PART II (b)

# Training 

# Different Probabilities 
p1 = [.4, .4, .2]
p2 = [.33, .33, .34]
p3 = [.8, .1, .1]

# Different Models w.r.t p1, p2, p3
clf1 = GaussianNB(priors=p1)
clf2 = GaussianNB(priors=p2)
clf3 = GaussianNB(priors=p3)

# Training the 3 Models
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

# Predicting with TEST data
y_with_test_pred1 = clf1.predict(X_test)
y_with_test_pred2 = clf2.predict(X_test)
y_with_test_pred3 = clf3.predict(X_test)

# Predicting with TRAINING data
y_with_training_pred1 = clf1.predict(X_train)
y_with_training_pred2 = clf2.predict(X_train)
y_with_training_pred3 = clf3.predict(X_train)

#### ACCURACY

# Accuracy with TEST data 

acc_test_p1 = accuracy_score(y_test, y_with_test_pred1)
acc_test_p2 = accuracy_score(y_test, y_with_test_pred2)
acc_test_p3 = accuracy_score(y_test, y_with_test_pred3)

# Accuracy with TRAINING data 

acc_training_p1 = accuracy_score(y_train, y_with_training_pred1)
acc_training_p2 = accuracy_score(y_train, y_with_training_pred2)
acc_training_p3 = accuracy_score(y_train, y_with_training_pred3)

#### Confusion Matrix

# Confusion Matrix with TEST data 

cm_test_p1 = confusion_matrix(y_test, y_with_test_pred1)
cm_test_p2 = confusion_matrix(y_test, y_with_test_pred2)
cm_test_p3 = confusion_matrix(y_test, y_with_test_pred3)

# Confusion Matrix with TRAINING data 

cm_training_p1 = confusion_matrix(y_train, y_with_training_pred1)
cm_training_p2 = confusion_matrix(y_train, y_with_training_pred2)
cm_training_p3 = confusion_matrix(y_train, y_with_training_pred3)

#### Printing on Console 

print(print_divider)
print("Accuracy Matrix for 3 classifier with TEST Data")
print(print_divider)
print(f"A) With Priors(40-40-20): {acc_test_p1}")
print(f"B) With Priors(33-33-34): {acc_test_p3}")
print(f"C) With Priors(80-10-10): {acc_test_p3}")

print(print_divider)
print("Accuracy Matrix for 3 classifier with TRAINING Data")
print(print_divider)
print(f"A) With Priors(40-40-20): {acc_training_p1}")
print(f"B) With Priors(33-33-34): {acc_training_p2}")
print(f"C) With Priors(80-10-10): {acc_training_p3}")

print(print_divider)
print("Confusion Matrix for 3 classifier with TEST Data")
print(print_divider)
print("A) With Priors(40-40-20)")
print(cm_test_p1)
print("B) With Priors(33-33-34)")
print(cm_test_p2)
print("C) With Priors(80-10-10)")
print(cm_test_p3)

print(print_divider)
print("Confusion Matrix for 3 classifier with TRAINING Data")
print(print_divider)
print("A) With Priors(40-40-20)")
print(cm_training_p1)
print("B) With Priors(33-33-34)")
print(cm_training_p2)
print("C) With Priors(80-10-10)")
print(cm_training_p3)

#### Saving the Accuracy data in a xls

# Saving the accuracy data in xls 

d = [[acc_test_p1, acc_test_p2, acc_test_p3], [acc_training_p1, acc_training_p2, acc_training_p3]]
c = ["P(40-40-20)", "P(33-33-34)", "P(80-10-10)"]
i = ["Accuracy(TEST Data)", "Accuracy(TRAINING Data)"]

acc_pd = pd.DataFrame(data=d, columns=c, index=i)

acc_pd.to_excel("accuracy_sheet.xls")
print(print_divider)
print("The ACCURACY of all the 3 different model w.r.t TEST and TRAINING data is saved in the file 'accuracy_sheet.xls'")
print(print_divider)

print(acc_pd)