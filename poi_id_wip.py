#!/usr/bin/python

from __future__ import division
import sys
import pickle
sys.path.append("../tools/")
import pprint
pp = pprint.PrettyPrinter(indent=4)

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#########
#My_funcs

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

def create_all_finance(df, nf):
    """
    Creates new feature by adding all financial features.
    Input Params:
    df: input dataset
	Output: updated dataset including new feature
    """
    fin_list = ['bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', \
	'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred', 'salary']
    
    for key in df:
        fin_sum = 0
        for f in fin_list:
            fin_sum = fin_sum + df[key][f]
        df[key][nf] = fin_sum

    return df

#########

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

# All features
features_list_all = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', \
'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', \
'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi', 'to_messages', 'total_payments', 'total_stock_value']

#features_list = features_list_all
#features_list = ['poi', 'prop_from_poi', 'prop_to_poi', 'all_finance']
#features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'exercised_stock_options', 'expenses', 'long_term_incentive', \
#'other', 'restricted_stock', 'salary', 'shared_receipt_with_poi', 'total_payments', 'total_stock_value', 'prop_from_poi', 'prop_to_poi']

features_list = ['poi', 'bonus', 'deferred_income', 'exercised_stock_options', 'expenses', 'other', 'restricted_stock', 'salary', \
'shared_receipt_with_poi', 'total_payments', 'total_stock_value', 'prop_from_poi', 'prop_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

d = {}
for person in data_dict:
    for key, value in data_dict[person].iteritems():
        if value == "NaN":
            if key in d:
                d[key] += 1
            else:
                d[key] = 1
#pp.pprint(d)

# Replace all NaN by zeros
data_dict = nan_2_zero(data_dict, features_list_all)

### Task 2: Remove outliers
# As already identified during the course curriculum (Outlier miniproject), remove TOTAL
# Remove outlier TOTAL
data_dict.pop( "TOTAL", 0 )
### Task 3: Create new feature(s)
data_dict = create_email_feature(data_dict, "from_poi_to_this_person", "to_messages", "prop_from_poi")
data_dict = create_email_feature(data_dict, "from_this_person_to_poi", "from_messages", "prop_to_poi")
#data_dict = create_all_finance(data_dict, "all_finance")
#pp.pprint(data_dict)
### Store to my_dataset for easy export below.
my_dataset = data_dict


#print "Datapoints in Dataset:"
#pp.pprint(len(my_dataset))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

import numpy as np
#print "Count of POIs in dataset:", np.sum(labels)
#pp.pprint(labels[144])
#pp.pprint(data[144])


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Playing with training datapoint resampling since it is an imbalanced dataset
#SMOTE
#from imblearn.over_sampling import SMOTE
#import numpy as np
#sm = SMOTE(random_state=42)
#features_train_res, labels_train_res = sm.fit_sample(features_train, labels_train)
#print (Y_train.value_counts() , np.bincount(y_train_res))

#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#print "Accuracy: ", accuracy_score(labels_test, pred)

#print "Important features: "
#pp.pprint(clf.feature_importances_)
#pp.pprint(data[144])

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

#clf = DecisionTreeClassifier()
#clf = DecisionTreeClassifier(min_samples_split=5)
#clf = RandomForestClassifier()
#clf = AdaBoostClassifier()
clf = AdaBoostClassifier(n_estimators=50)
clf.fit(features_train, labels_train)

# Fitting resampled training set
#clf.fit(features_train_res, labels_train_res)

#pred = clf.predict(features_test)
#print "Accuracy: ", accuracy_score(labels_test, pred)
#print "Feature importance: "
pp.pprint(clf.feature_importances_)

# Get the importance of each feature from feature_importances_
#pp.pprint(feature_importances(data_dict, features_list, 0.3))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
