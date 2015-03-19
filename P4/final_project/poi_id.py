#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from selection import select_k_best_features, best_parameter_from_search, precision_n_recall
from calculation import computeFraction

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Get the total number of pois in the dataset
num_poi = 0
for name in data_dict.keys():
    if data_dict[name]['poi'] == True:
        num_poi += 1

print "There are", num_poi, "POIs in total."
### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
    data_dict.pop(outlier, 0)



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Create new features
for name in my_dataset:
    my_point = my_dataset[name]
    ### fraction of emails one person got from poi
    my_point["fraction_from_poi"] = \
    computeFraction( my_point["from_poi_to_this_person"], my_point["to_messages"] )
    ### fraction of emails one person sent to poi
    my_point["fraction_to_poi"] = \
    computeFraction( my_point["from_this_person_to_poi"], my_point["from_messages"] )
    ### number of poi as shared recipients per received email 
    my_point["shared_poi_per_email"] = \
    computeFraction( my_point["shared_receipt_with_poi"], my_point["to_messages"] )
    ### indictor of whether a person has an email address
    if my_point['email_address'] == 'NaN':
        my_point['email_exists'] = 0
    else:
        my_point['email_exists'] = 1


### Full list of features (always starts with 'poi' -- the value to predict), 
### including the both original and newly created ones, expect for 'email_adress', 
### which is more of an id rather than a feature.
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', \
                 'loan_advances', 'bonus', 'restricted_stock_deferred', \
                 'deferred_income', 'total_stock_value', 'expenses', \
                 'exercised_stock_options', 'other', 'long_term_incentive', \
                 'restricted_stock', 'director_fees', 'to_messages', \
                 'email_exists', 'fraction_from_poi', 'fraction_to_poi', \
                 'shared_poi_per_email','from_messages',
                 'shared_receipt_with_poi']


### Naive Bayesian Classifier with 8 Best Features

def callNBC():
    ### Feature selection
    num_to_keep = 8
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)
    clf = GaussianNB()

    tester_prep(clf, my_dataset, my_features_list)

### Random Forest Classifier with best K features
def callRFC():
    ### Feature selection
    num_to_keep = 9
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)

    ### Make classier pipeline
    pipeline_rfc = Pipeline(steps=[
        ('classifier', RandomForestClassifier(random_state = 42))  
    ])

    parameters_rfc = {
        'classifier__max_features': ('sqrt', 1),
        'classifier__max_depth': np.arange(3, 8),
        'classifier__n_estimators' : (10, 20)
    }

    ### Grid search for the optimal parameters
    precision_n_recall(pipeline_rfc, parameters_rfc, my_dataset, my_features_list)

    clf = RandomForestClassifier(max_depth = 5, 
                                 max_features = 'sqrt', 
                                 n_estimators = 10, 
                                 random_state = 42)

    tester_prep(clf, my_dataset, my_features_list)


### Logistic Regression with best K features
def callLR():
    ### Feature selection
    num_to_keep = 16
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)

    ### Make classier pipeline
    pipeline_lrg = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(tol = 0.001, random_state = 42))
    ])

    parameters_lrg = {
        'classifier__penalty': ('l1', 'l2'),
        'classifier__C': 10.0 ** np.arange(-10, -3)
        }

    ### Grid search for the optimal parameters
    precision_n_recall(pipeline_lrg, parameters_lrg, my_dataset, my_features_list)

    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', 
                                              random_state = 42))
    ])

    tester_prep(clf, my_dataset, my_features_list)


### Support Vector Classifier with K Best Features
def callSVC():
    ### Feature selection
    num_to_keep = 8
    my_features_list = select_k_best_features(my_dataset, features_list, num_to_keep)

    ### Make classier pipeline
    pipeline_svc = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel = 'rbf', random_state = 42, class_weight = 'auto'))
    ])

    parameters_svc = {
        'classifier__gamma': 10.0 ** np.arange(-4, 0),
        'classifier__C': 10.0 ** np.arange(1, 5)
        }


    ### Grid search for the optimal parameters
    precision_n_recall(pipeline_svc, parameters_svc, my_dataset, my_features_list)

    clf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel = 'rbf', C = 1000, gamma = 0.0001, 
                               random_state = 42, class_weight = 'auto'))
    ])

    tester_prep(clf, my_dataset, my_features_list)


def tester_prep(clf, my_dataset, my_features_list):
    print("Testing the performance of the classifier...")
    test_classifier(clf, my_dataset, my_features_list)

    ### Dump classifier, dataset, and features_list so anyone can run/check the results.
    dump_classifier_and_data(clf, my_dataset, my_features_list)


# Main Function
def main(argv):
    method = int(sys.argv[1])

    classifier_dictionary = \
    {1: 'Naive Bayesian Classifier', 
     2: 'Random Forest Classifier', 
     3: 'Logistic Regression', 
     4: 'Support Vector Classifier'}

    options = {1: callNBC, 2: callRFC, 3: callLR, 4: callSVC} 

    print "Let's get started with " + str(classifier_dictionary[method]) + "!!"
    options[method]()   
         
if __name__ == "__main__":
    main(sys.argv[1:])