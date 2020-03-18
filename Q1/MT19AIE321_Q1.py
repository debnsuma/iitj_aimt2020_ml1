# Author : Suman Debnath
# Email : debnath.1@iitj.ac.in
# Roll No : MT19AIE321
# M.Tech-AI(2020) 
# Date : 16th March 2020

#################################
#  Solution for Question No. 1  #
#################################

# Importing all the modules 
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import multilabel_confusion_matrix
from itertools import cycle
from pprint import pprint

print_divider = "*"* 130

def entropy(target_col):

    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])

    return entropy

def infoGain(data,split_attribute_name,target_name="class_label"):

    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy

    return Information_Gain
       
def id3_algo(data,originaldata,features,target_attribute_name="class_label",parent_node_class = None):

    #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
    
    #If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    #If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    #the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    #the mode target feature value is stored in the parent_node_class variable.
    
    elif len(features) ==0:
        return parent_node_class
    
    #If none of the above holds true, grow the tree!
    
    else:
        #Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        item_values = [infoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {best_feature:{}}
        
        
        #Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = id3_algo(sub_data,originaldata,features,target_attribute_name,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return tree    
                
def predict(query,tree,default = 1):
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result

def my_train_test_split(dataset, split_ratio=.80):
    
    total_rows = dataset.shape[0]
    split = int(split_ratio * total_rows)
    
    training_data = dataset.iloc[:split].reset_index(drop=True)    
    testing_data = dataset.iloc[split:].reset_index(drop=True)
    
    return training_data,testing_data

def test(data,tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    # print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class_label"])/len(data))*100,'%')

    return predicted

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

def sklearn_dt_classifier(X_train, y_train, crit='gini'):
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
    if algo == "DT_CUSTOM":
        plt.title('ROC for Custom Decision Tree')
    elif algo == "DT_SCIKIT":
        plt.title('ROC for Scikit Learn Decision Tree')
    else:
        plt.title('ROC for Decision Tree')
    plt.legend(loc="lower right")

    plt.savefig(image_name)

    roc_for_algo["roc_auc"] = roc_auc
    roc_for_algo["tpr"] = tpr
    roc_for_algo["fpr"] = fpr

    return roc_for_algo

def main():   

    # Taking the input data set file and validation of the file 
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
    dataset = pd.read_csv(input_file, delimiter="\t", header=None)

    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset.columns = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "class_label"]
    # Create X (Input features)
    X = dataset.drop("class_label", axis=1)
    # Create y (Class labels)
    Y = dataset["class_label"]

    # Spliting the dataset 
    print(print_divider)
    training_data, testing_data = my_train_test_split(dataset)
    X_train = training_data.drop("class_label", axis=1)
    X_test = testing_data.drop("class_label", axis=1)
    y_train = training_data["class_label"]
    y_test = testing_data["class_label"]

    ################### CUSTOM DECISION TREE CLASSIFIER #############################
    # Training the Custom DT 
    print(print_divider)
    print("Training the Custom Decision Tree")
    tree = id3_algo(training_data,training_data,training_data.columns[:-1])

    # Printing the Custom DT
    print("1. CUSTOM DECISION TREE CLASSIFIER")
    print(print_divider)
    pprint(tree)

    # Predecting the result from Custom DT
    print(print_divider)
    predicted_custom_dt = test(testing_data,tree)

    # Evaluating the Performance of the Custom DT model
    y_pred_custom = predicted_custom_dt["predicted"].to_numpy()
    y_pred_custom = y_pred_custom.astype(int)
    print("1. Model Evaluation for 'CUSTOM DECISION TREE CLASSIFIER'")
    model_evaluation(y_test, y_pred_custom)

    # ROC Curve for Decision Tree 
    roc_for_dt = roc_curve_generator(Y=Y, y_test=y_test, y_pred=y_pred_custom, image_name="ROC_CURVE_CUSTOM_DT_Q2.png", algo="DT_CUSTOM")
    roc_dt_img = os.path.abspath("ROC_CURVE_CUSTOM_DT_Q2.png")
    print(print_divider)

    ################### SCIKIT LEARN DECISION TREE CLASSIFIER #############################
    
    # Training with Scikit-Learn DT 
    print("2. SCIKIT LEARN DECISION TREE CLASSIFIER")
    clf_tree = sklearn_dt_classifier(X_train, y_train, crit='gini')

    # Predecting the result with Scikit-Learn DT 
    y_pred = clf_tree.predict(X_test)

    # Evaluating the Performance of the Scikit-Learn DT 
    print("2. Model Evaluation for 'SCI-KIT LEARN DECISION TREE CLASSIFIER'")
    model_evaluation(y_test, y_pred)

    # ROC Curve for Decision Tree 
    roc_for_dt = roc_curve_generator(Y=Y, y_test=y_test, y_pred=y_pred, image_name="ROC_CURVE_SKLEARN_DT_Q2.png", algo="DT_SCIKIT")
    roc_dt_img = os.path.abspath("ROC_CURVE_SKLEARN_DT_Q2.png")

    # Location of the ROC Curve
    print(f"ROC Curve for Custom Decision Tree is saved at : {roc_dt_img}")
    print(f"ROC Curve for Scikit Learn Decision Tree is saved at : {roc_dt_img}")
    print(print_divider)

if __name__ == "__main__":
    main()