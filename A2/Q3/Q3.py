# Author : Suman Debnath
# Email : debnath.1@iitj.ac.in
# Roll No : MT19AIE321
# M.Tech-AI(2020) 
# Date : 10th April 2020

#################################
#  Solution for Question No. 3  #
#################################

# Importing the modules 
import pandas as pd
import numpy as np
import operator

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, roc_auc_score

from sklearn.model_selection import cross_val_score
from pprint import pprint as pp
from scipy.stats import zscore

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

def split_data_set(X, y):
    
    X_train, X_remianing, y_train, y_remaining = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_remianing, y_remaining, test_size=.33, random_state=42)
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def KNN_Classifier(n_neighbors=5, metric='minkowski', weights='uniform', p=2):
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p, metric=metric)
    
    return clf

def hpyer_combination_generator():
    
    # (a) Three different values of k (no. of neighbors).
    k_values = [3, 7, 11]
    
    #(b) Two different distance metrics (Minskowski and Euclidean).
    distance_values = ["minkowski", "euclidean"]
    
    #(c) All points in each neighborhood weighted equally.
    #(d) Points in the neighborhood weighted by their distance such that the closer 
    #    neighbors of a query point have a greater influence than neighbors which are farther away.
    weight_values = ["uniform", "distance"]

    hyper_prams_combi = []

    for k in k_values:
        for d in distance_values:
            if d == "minkowski":
                p = 1
            else:
                p = 2
            for w in weight_values:
                _ = [k, d, w, p]
                hyper_prams_combi.append(_)
    return hyper_prams_combi

def all_clf(hyper_prams):
    all_clf_dict = {}
    for h_pram in hyper_prams:
        name = f"{h_pram[0]}_{h_pram[1]}_{h_pram[2]}"
        n, m, w, p = h_pram[0], h_pram[1], h_pram[2], h_pram[3]
        all_clf_dict[name] = KNN_Classifier(n_neighbors=n, metric=m, weights=w, p=p)
    
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


print_divider = "*"* 100
## Loading the dataset
data = load_breast_cancer()
data_array = data['data']
target_array = data['target']
data_column_name = data['feature_names']
data_target_name = data['target_names']

X = pd.DataFrame(data['data'], columns=data['feature_names'])
Y = pd.Series(target_array)
# Normalizing the data
X = X.apply(zscore)

## Spliting the data into training, test and validation
X_train, X_test, X_val, y_train, y_test, y_val = split_data_set(X, Y)

## Training with KNN

### Classifier accuracies on `TEST` data set
hyper_prams = hpyer_combination_generator()
all_clf_dict = all_clf(hyper_prams)
clf_fit(all_clf_dict, X_train, y_train)
clf_preds = clf_preditions(X_test, all_clf_dict)

clf_accuracy_score_for_test_data = clf_accuracy_score(y_test, clf_preds)

print(print_divider)
print("Report classifier accuracy on the `TEST` data set")
print(print_divider)
print(clf_accuracy_score_for_test_data)

### Classifier accuracies on `VALIDATION` data set
hyper_prams = hpyer_combination_generator()
all_clf_dict = all_clf(hyper_prams)
clf_fit(all_clf_dict, X_train, y_train)
clf_preds = clf_preditions(X_val, all_clf_dict)

clf_accuracy_score_for_val_data = clf_accuracy_score(y_val, clf_preds)

print(print_divider)
print("Report classifier accuracy on the `VALIDATION` data set")
print(print_divider)
print(clf_accuracy_score_for_val_data)

### Classifier accuracies on `TRAIN` data set
hyper_prams = hpyer_combination_generator()
all_clf_dict = all_clf(hyper_prams)
clf_fit(all_clf_dict, X_train, y_train)
clf_preds = clf_preditions(X_train, all_clf_dict)

clf_accuracy_score_for_train_data = clf_accuracy_score(y_train, clf_preds)

print(print_divider)
print("Report classifier accuracy on the `TRAIN` data set")
print(print_divider)
print(clf_accuracy_score_for_train_data)

## PART II

### Model Evaluation (`TRAIN` and `VALIDATION` set)
plt.figure(figsize=(10,10))
plt.plot(*zip(*sorted(clf_accuracy_score_for_train_data.items())), label="Train Data")
plt.plot(*zip(*sorted(clf_accuracy_score_for_val_data.items())), label="Validation Data")
plt.xlabel('Different Classifiers')
plt.ylabel('Model Accuracy')
plt.title('Model Evaluations')
plt.ylim(0, 1.1)
plt.grid(axis='x', color='0.95')
plt.legend()
plt.xticks(rotation=90)

#plt.show()
plt.savefig("Model_Evaluations_Train_Val.png")
print(print_divider)
print("Model Evaluation (`TRAIN` and `VALIDATION` set) is saved in: 'Model_Evaluations_Test_Val.png'")
print(print_divider)

### Identify the top-two classifiers on the `VALIDATION` set.

clf_accuracy_score_for_val_data_sorted = dict(sorted(clf_accuracy_score_for_val_data.items(), \
                                                     key=operator.itemgetter(1), reverse=True))
print("Report of all classifier accuracy on the `VALIDATION` data set")
print(clf_accuracy_score_for_val_data_sorted)
print(print_divider)

### Report the accuracy on the `TEST` set

clf_accuracy_score_for_test_data_sorted = dict(sorted(clf_accuracy_score_for_test_data.items(), \
                                                     key=operator.itemgetter(1), reverse=True))

print("Report of all classifier accuracy on the `TEST` data set")
print(clf_accuracy_score_for_test_data_sorted)
print(print_divider)

plt.figure(figsize=(10,10))
plt.plot(*zip(*sorted(clf_accuracy_score_for_test_data_sorted.items())), label="Test Data")
plt.xlabel('Different Classifiers')
plt.ylabel('Model Accuracy')
plt.title('Model Evaluations')
plt.ylim(0, 1.1)
plt.grid(axis='x', color='0.95')
plt.legend()
plt.xticks(rotation=90)

#plt.show()
plt.savefig("Model_Evaluations_Test.png")
print("Model Evaluation with TEST dataset is saved in: 'Model_Evaluations_Test.png'")
print(print_divider)

### Accuracy of these two best classifiers using VALIDATION DataSet 
_tmp = list(clf_accuracy_score_for_val_data_sorted.keys())
first_best, second_best = _tmp[0], _tmp[1]

print(f" Accuracy of best classifier 1 ({first_best}): {clf_accuracy_score_for_val_data_sorted[first_best]}")
print(f" Accuracy of best classifier 2 ({second_best}): {clf_accuracy_score_for_val_data_sorted[second_best]}")

print(print_divider)


## PART III

### Use both training and validation split for training.

X_train_new = pd.concat([X_train, X_val], axis=0, sort=False)
y_train_new = pd.concat([y_train, y_val], axis=0, sort=False)
### Use only the first two features of the dataset for training
X_train_with_two_feature = X_train_new[X_train_new.columns[:2]]
X_test_with_two_feature = X_test[X_test.columns[:2]]

### Best two classifiers
'''
- n_neighbors=3, metric="minkowski", weights="uniform", p=1
- n_neighbors=3, metric="minkowski", weights="distance", p=1

'''

best_clf_1 = KNN_Classifier(n_neighbors=3, metric="minkowski", weights="uniform", p=1)
best_clf_2 = KNN_Classifier(n_neighbors=3, metric="minkowski", weights="distance", p=1)

best_clf_1.fit(X_train_with_two_feature, y_train_new)
best_clf_2.fit(X_train_with_two_feature, y_train_new)

clf_pred_best_clf_1_on_training = best_clf_1.predict(X_test_with_two_feature)
clf_pred_best_clf_2_on_training = best_clf_2.predict(X_test_with_two_feature)

print(f"Now building a new classifier with the above two best classifier '{first_best}' and '{second_best}', using only the first 2 features")
print(print_divider)

### Plot decision boundaries obtained and compare them.

# Converting the df to numpy array 
X = X_train_with_two_feature.to_numpy()
y = y_train_new.to_numpy()
X2_test = X_test_with_two_feature.to_numpy()
y2_test = y_test.to_numpy()

n_neighbors = 3
metric="minkowski" 
p = 1

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = KNN_Classifier(n_neighbors=n_neighbors, metric=metric, weights=weights, p=p)
    clf.fit(X, y)

    clf_pred_train = clf.predict(X)
    acc = accuracy_score(clf_pred_train, y)
    print(f"Accuracy (k = {n_neighbors}, weights = {weights}, metric = {metric}) with TRAIN Data Set is: {acc}")
    
    clf_pred_test = clf.predict(X2_test)
    acc2 = accuracy_score(clf_pred_test, y2_test)
    print(f"Accuracy (k = {n_neighbors}, weights = {weights}, metric = {metric}) with TEST Data Set is: {acc2}")

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,10))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"2-Class classification (k = {n_neighbors}, weights = {weights}, metric = {metric})")

    #plt.show()
    print(f"Decision boundary obtained with (k={n_neighbors}, weights={weights} and metric={metric})")
    f_name = f"boundary_plot_{n_neighbors}_{weights}_{metric}.png"
    plt.savefig(f_name)
    print(f"The decision boundary plot is saved at: {f_name}")
    print(print_divider)
