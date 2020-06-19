#!/usr/bin/python

from __future__ import division
import sys
import pickle
sys.path.append("../tools/")
import pprint

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### My_functions

def nan_2_zero(df, fl):
    """
    All NaN values replaced by zero.
	Input Params:
    df: input dataset
    fl: feature list
	Output: Updated dataset (df) with all NaN replaced by 0.
    """
    for f in fl:
        for key in df:
            if df[key][f] == "NaN":
                df[key][f] = 0

    return df

def create_email_feature(df, f1, f2, nf):
    """
    Creates a new feature from existing features.
    Input Params:
    df: input dataset
    f1 or f2: existing features 
    nf: new feature created (f1 divide by f2)
    Output: Updated dataset (df) with newly added feature.
    """
    for key in df:
        if df[key][f1] == 0:
            df[key][nf] = 0.0
        elif df[key][f2] == 0:
            df[key][nf] = 0.0
        else:
            df[key][nf] = float(df[key][f1]) / float(df[key][f2])

    return df

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### All features
features_list_all = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', \
'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', \
'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi', 'to_messages', 'total_payments', 'total_stock_value']

features_list = ['poi', 'bonus', 'deferred_income', 'exercised_stock_options', 'expenses', 'other', 'restricted_stock', 'salary', \
'shared_receipt_with_poi', 'total_payments', 'total_stock_value', 'prop_from_poi', 'prop_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Replace all NaN by zeros
data_dict = nan_2_zero(data_dict, features_list_all)


### Task 2: Remove outliers
### As already identified during the Outlier miniproject, remove TOTAL
### Remove outlier TOTAL
data_dict.pop( "TOTAL", 0 )

### Task 3: Create new feature(s)
data_dict = create_email_feature(data_dict, "from_poi_to_this_person", "to_messages", "prop_from_poi")
data_dict = create_email_feature(data_dict, "from_this_person_to_poi", "from_messages", "prop_to_poi")


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50, random_state=42)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import accuracy_score
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "Accuracy: ", accuracy_score(labels_test, pred)
print "Feature importance: "
print clf.feature_importances_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
