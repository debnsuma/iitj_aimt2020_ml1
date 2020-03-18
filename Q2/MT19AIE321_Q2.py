# Author : Suman Debnath
# Email : debnath.1@iitj.ac.in
# Roll No : MT19AIE321
# M.Tech-AI(2020) 
# Date : 16th March 2020

#################################
#  Solution for Question No. 2  #
#################################

# Importing all the modules 
import os
import sys
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, roc_auc_score
from sklearn import preprocessing
from itertools import cycle

# Setting a RANDOM SEED for consistant result(deterministic random data) 
RANDOM_SEED=13
print_divider = "*"* 130

def load_data(file_name):
    '''
    file_name: Full path of the input file(data set)
    returns: pandas dataframe 
    '''
    data_df = pd.read_csv(file_name, header = None)
    column_header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    data_df.columns = column_header

    # Encoding the class label 
    le = preprocessing.LabelEncoder()
    data_df["class"] = le.fit_transform(data_df['class'])

    return data_df, le 

def split_data(data_df, test_ratio=0.20):
    '''
    data_df: Input dataframe with the Class Labels
    return: 
        A tuple which consists of-
            X_train - Training Data
            X_test - Testing Data
            y_train - Training Class Label 
            y_test - Testing Class Label 
    '''
    # Seperating the class label and the input features 
    # Create X (Input features)
    X = data_df.drop("class", axis=1)
    # Create y (Class labels)
    Y = data_df["class"]

    # Spliting the dataset into training and testing data set 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio, random_state = RANDOM_SEED)

    return (X_train, X_test, y_train, y_test, X, Y)

def dt_classifier(X_train, y_train, crit='gini'):
    '''
    X_train: Input features 
    y_train: Class label for the training data(X_train)
    criterion: default is "gini", it could be "entropy" as well

    returns:
    clf_tree: df_classifier object 
    '''
    clf_tree = DecisionTreeClassifier(criterion=crit)
    clf_tree.fit(X_train, y_train)

    return clf_tree

def rf_classifier(X_train, y_train, n_estimators=10):
    '''
    X_train: Input features 
    y_train: Class label for the training data(X_train)
    n_estimators: default is 10

    returns:
    clf_tree: rf_classifier object 
    '''
    clf_tree = RandomForestClassifier(n_estimators=10)
    clf_tree.fit(X_train, y_train)

    return clf_tree

def model_evaluation(y_test, y_pred):
    # Accuracy 
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion Matrix 
    cm = confusion_matrix(y_test, y_pred)

    print(print_divider)
    # Precision and Recall (Classification Report) 
    clf_report = classification_report(y_test, y_pred)
    print("Classification Report")
    print(clf_report)
    print(print_divider)

    # Sensitivity and Specificity
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    sensitivity = tp / (tp + fn) 
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)

    print(f"Accuracy : {accuracy}")
    print(print_divider)
    print(f"Confusion Matrix : \n {cm}")
    print(print_divider)
    for c in range(len(precision)):
        print(f"Precision for class {c} : {precision[c]}")
    print(print_divider)
    for c in range(len(sensitivity)):
        print(f"Sensitivity for class {c} : {sensitivity[c]}")
    print(print_divider)
    for c in range(len(specificity)):
        print(f"Specificity for class {c} : {specificity[c]}")
    print(print_divider)
    
def roc_curve_generator(Y, y_test, y_pred, image_name, algo):

    # Total no. of unique class 
    n_classes = len(Y.unique())

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    roc_for_algo = dict() 

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Final averaging it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot the ROC curves
    line_width=3
    plt.figure(figsize=(10,10))
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='green', linestyle=':', lw=line_width)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=line_width,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--',color='green', lw=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    if algo == "DT":
        plt.title('ROC for Decision Tree')
    elif algo == "RF":
        plt.title('ROC for Random Forest')
    else:
        plt.title('ROC - IRIS Dataset ')
    plt.legend(loc="lower right")

    plt.savefig(image_name)

    roc_for_algo["roc_auc"] = roc_auc
    roc_for_algo["tpr"] = tpr
    roc_for_algo["fpr"] = fpr

    return roc_for_algo

def roc_compare_between_model(dt, rf, Y):

    roc_auc = dt["roc_auc"]
    roc_auc2 = rf["roc_auc"]
    fpr = dt["fpr"]
    fpr2 = rf["fpr"] 
    tpr = dt["tpr"]
    tpr2 = rf["tpr"] 

    line_width=2

    # Total no. of unique class 
    n_classes = len(Y.unique())

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of Class {i} with DT(area = %0.2f)' % roc_auc[i], lw=line_width)
        plt.plot(fpr2[i], tpr2[i], label=f'ROC curve of Class {i} with RF(area = %0.2f)' % roc_auc2[i], lw=line_width)
        plt.plot([0, 1], [0, 1], 'k--',color='green', lw=line_width)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve between DT and RF for class label {i}')
        plt.legend(loc="lower right")
        roc_img_file_name = f"ROC_DT_vd_RF_class_label_{i}.png"
        plt.savefig(roc_img_file_name)

        img = os.path.abspath(roc_img_file_name)
        print(f"ROC Curve between DT and RF for class label {i} is saved at : {img}")
        print(print_divider)

    # Plot and visualize your Decision Tree 

def roc_scores(clf, X_test, y_test):

    y_prob = clf.predict_proba(X_test)

    macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted")
    macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",average="macro")
    weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print("One-vs-One ROC AUC scores:\n{:.4f} (macro),\n{:.4f} "
        "(weighted by prevalence)"
        .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    print("One-vs-Rest ROC AUC scores:\n{:.4f} (macro),\n{:.4f} "
        "(weighted by prevalence)"
        .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

def dt_graph(clf_tree, data_df, le):

    dot_data = tree.export_graphviz(clf_tree, out_file=None, 
                        feature_names=data_df.columns[:-1],  
                        class_names=le.classes_,  
                        filled=True, rounded=True,  
                        special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render(filename='DT_Graph')
    f_name = os.path.abspath('DT_Graph.pdf')
    print(f"Visual Graph of the Decision Tree is saved at : {f_name}")
    print(print_divider)

def main():

    # Taking the input data set file and validation of the file 
    print(print_divider)
    print("Checking for the input file path..")
    if len(sys.argv) == 2: 
        input_file = sys.argv[1]
        input_file = os.path.abspath(input_file)
        if os.path.isfile(input_file):
            print("Checking for the input file path...DONE")
            print(print_divider)
            print(f"Full path of your data set is : {input_file}")
            print(print_divider)
        else:
            print(print_divider)
            print(f"Invalide file or file path {input_file}")
            sys.exit(2)
    else: 
        print(print_divider)
        print("Please provide a valid file path to the file..")
        sys.exit(2)
    
    # Loading the dataset in the dataframe 
    print("Loading the data set...")
    data_df, le = load_data(input_file)
    print("Loading the data set...DONE")
    print(print_divider)

    # Splitting the dataframe (trainng and testing)
    print("Splitting the data set into training and test...")
    X_train, X_test, y_train, y_test, X, Y = split_data(data_df, test_ratio=.20)
    print("Splitting the data set into training and test...DONE")
    print(print_divider)
    print(f"Total Data Points: {data_df.shape[0]}")
    print(f"No. of Data Points(Training): {X_train.shape[0]}")
    print(f"No. of Data Points(Testing): {X_test.shape[0]}")
    print(print_divider)

    ################### Decision Tree #############################
    print("1. DECISION TREE CLASSIFIER")
    print(print_divider)
    # Training with Decision Tree Classifier 
    clf_tree = dt_classifier(X_train=X_train, y_train=y_train)
    print("Hyper Parameters used:")
    for pram in clf_tree.get_params():
        print(f"{pram} : {clf_tree.get_params()[pram]}")
    print(print_divider)

    # Perform the prediction on the test data 
    print("Performing the prediction on the test data...")
    y_pred = clf_tree.predict(X_test)
    print("Performing the prediction on the test data...DONE")

    # Evaluating the Performance of the model 
    model_evaluation(y_test, y_pred)

    # Build the Confusion Matrix 
    plot_confusion_matrix(clf_tree, X_test, y_test)
    
    # ROC Curve for Decision Tree 
    roc_for_dt = roc_curve_generator(Y=Y, y_test=y_test, y_pred=y_pred, image_name="ROC_CURVE_DT.png", algo="DT")
    roc_dt_img = os.path.abspath("ROC_CURVE_DT.png")
    print(f"ROC Curve for Decision Tree is saved at : {roc_dt_img}")
    print(print_divider)

    ################### Random Forest #############################
    print("2. RANDOM FOREST CLASSIFIER")
    print(print_divider)
    # Training with Random Forest Classifier 
    clf_tree2 = rf_classifier(X_train=X_train, y_train=y_train, n_estimators=10)
    print("Hyper Parameters used:")
    for pram in clf_tree2.get_params():
        print(f"{pram} : {clf_tree2.get_params()[pram]}")
    print(print_divider)

    # Perform the prediction on the test data 
    print("Performing the prediction on the test data...")
    y_pred2 = clf_tree2.predict(X_test)
    print("Performing the prediction on the test data...DONE")

    # Evaluating the Performance of the model 
    model_evaluation(y_test, y_pred2)

    # Build the Confusion Matrix 
    plot_confusion_matrix(clf_tree2, X_test, y_test)
    
    # ROC Curve for Random Forest 
    roc_for_rf = roc_curve_generator(Y=Y, y_test=y_test, y_pred=y_pred2, image_name="ROC_CURVE_RF.png", algo="RF")
    roc_rf_img = os.path.abspath("ROC_CURVE_RF.png")
    print(f"ROC Curve for Random Forest is saved at : {roc_rf_img}")
    print(print_divider)

    # ROC Curve Comparision between Decision Tree and Random Forest Classifier for all the class labels 
    roc_compare_between_model(dt=roc_for_dt, rf=roc_for_rf, Y=Y)
    print(print_divider)

    # Plot and visualize your Decision Tree
    dt_graph(clf_tree, data_df, le)

    # ROC AUC Scores for Decision Tree 
    print("ROC AUC Scores for Decision Tree")
    roc_scores(clf_tree, X_test, y_test)
    print(print_divider)

    # ROC AUC Scores for Random Forest 
    print("ROC AUC Scores for Random Forest")
    roc_scores(clf_tree2, X_test, y_test)
    print(print_divider)

    print("Check all the graphs in the present working directory")
    print(print_divider)

if __name__ == '__main__':
    main() 