# Identify-Fraud-from-Enron-Email

Udacity Project on Introduction to Machine Learning for AIRBUS Data Analyst Nanodegree Program.

**Title**: Identify Fraud from Enron Email  
**Author**: Pankaj NATH

`File Name`|`Issue`|`Date`|`Reason for Revision`
-----------|-------|------|---------------------
poi_id.py|v1.0|20th Jun, 2020|First submission
my_features_list.pkl|v1.0|20th Jun, 2020|First submission
my_dataset.pkl|v1.0|20th Jun, 2020|First submission
my_classifier.pkl|v1.0|20th Jun, 2020|First submission
Refrences.txt|v1.0|20th Jun, 2020|First submission
Response Questions.pdf|v1.0|20th Jun, 2020|First submission
  
## Summary  
  
This repository contains files for final Udacity project submission for Introduction to Machine Learning.  
A ML classifier which uses the features provided to classify a person as poi (person-of-interest) for fraud.  
  
The tester function gives below result:
  
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=42)
**Accuracy**: 0.85087
**Precision**: 0.42418      
**Recall**: 0.33150           
**F1**: 0.37216                  
**F2**: 0.34665
**Total predictions**: 15000        
**True positives**: 663    
**False positives**: 900     
**False negatives**: 1337        
**True negatives**: 12100
  
  
## File List and Description
* poi_id.py : The main python script.
* my_features_list.pkl : Pickle file generated after running **poi_id.py** and contains the features used by the classifier.
* my_dataset.pkl : Pickle file generated after running **poi_id.py** and contains the complete dataset with newly created features as well.
* my_classifier.pkl : Pickle file generated after running **poi_id.py** and contains the trained classifier.
* Refrences.txt : Text file with all the resources referred during this project work.
* Response Questions.pdf : Free response/answers to the questions as part of this project submission.
