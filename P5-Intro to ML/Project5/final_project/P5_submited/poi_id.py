#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier  
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from sklearn.tree import DecisionTreeClassifier
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
sys.path.append("../tools/")


# define function: dataset summary 
def dateset_summary():
	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)
	n_poi,n_non_poi = 0,0
	for key in data_dict.keys():
		if data_dict[key]['poi']==1:
			n_poi+=1
		else:
			n_non_poi+=1        
	print("Total number of data points: ", len(data_dict))
	print("Number of POI: %d, no. of non-POI: %d"% (n_poi, n_non_poi))
	print("Obviously, there is a class imbalance Problem.")
	print("Number of features used: ",len(data_dict['METTS MARK'])) # randomly pick one name, get the number of features.
	return data_dict



### function: Remove outliers
def remove_outerliers(data_dict,names):
	for name in names:
		data_dict.pop(name, 0)
	return data_dict


### function: make new feature 
#1. email_features_miss: dummy variable, if email_features is missing.   
#2. poi_rate_to_messages = from_this_person_to_poi/to_messages  
#3. poi_rate_from_messages = from_poi_to_this_person/from_messages

def add_new_features(data_df):
	data_df['email_features_miss'] = data_df.apply(lambda x: 1 if x["to_messages"]=='NaN' else 0, axis = 1)
	data_df['poi_rate_to_messages'] = data_df.apply(lambda x: 0 if x['from_this_person_to_poi']=='NaN' \
                                                else x['from_this_person_to_poi']*1.0/x['to_messages'], axis=1)
	data_df['poi_rate_from_messages'] = data_df.apply(lambda x: 0 if x['from_poi_to_this_person']=='NaN'\
                                                  else x['from_poi_to_this_person']*1.0/x['from_messages'], axis =1)
	return data_df



### Store to my_dataset,features_list 
def full_dataset_fatures(data_df):
	features_list = list(data_df.columns.values)
	features_list.remove('email_address')
	features_list.remove('poi')
	features_list = ['poi'] +features_list
	my_dataset = data_df.to_dict(orient='index')
	print ("%d features.\n columns_name: %s" % (len(features_list), features_list))	
	return my_dataset, features_list



### function: Extract features and labels from dataset 
def get_features_labels(my_dataset,features_list):
	data = featureFormat(my_dataset, features_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)
	features = np.array(features)
	labels = np.array(labels)
	return features,labels


### fucntion: define an evaluation metric 
"""
As precision and recall of final classifier are both supposed no less than 0.3, I define an evaluation metric that take both into account. 
This metric returns f1_score, if precision and recall both over 0.3; returns 0.3 if only one overs 0.3;  returns 0 if neiher of them over 0.3.
"""

def scorer_r_p(estimator, features_test, labels_test):
	labels_pred = estimator.predict(features_test)
	pre= precision_score(labels_test, labels_pred, average='micro')
	rec = recall_score(labels_test, labels_pred, average='micro')
	if pre>0.3 and rec>0.3:
		return f1_score(labels_test, labels_pred, average='macro')
	elif  pre>0.3 and rec<0.3:
		return 0.3
	elif rec >0.3 and pre<0.3:
		return 0.3
	return 0



# function: build model and tuning parameters
def train_classifier(features,labels,n=50):
    # Stratified ShuffleSplit cross-validator
	sss = StratifiedShuffleSplit(labels, n_iter=n, test_size = 0.2, random_state=42)
	# Pipeline of transforms with a final estimator.
	pipe = Pipeline([('scaler',MinMaxScaler(feature_range=(0,1))), 
	                ('selector', SelectKBest()),
	                  ('classifier', AdaBoostClassifier(random_state=32))])
	# Specify a parameter grid.
	# The tuned parameters are k (number of selected features) and score function in feature selection process,
	# iteration times , learning rate, base estimator (min_samples_split of DecisionTree) of Adaptive Boosting algorithm.
	param_grid = dict( selector__score_func=[chi2,f_classif]
	                 ,selector__k=range(4,11)
	                 ,classifier__n_estimators=[50,80,100]
	                 ,classifier__learning_rate=[0.5,1,1.5,2]
	                 ,classifier__base_estimator=[DecisionTreeClassifier(min_samples_split=2)\
	                                                , DecisionTreeClassifier(min_samples_split=3)\
	                                                , DecisionTreeClassifier(min_samples_split=4)]\
	                 )
	# Exhaustive search over specified parameter values for AdaBoostClassifier.
	clf_grid = GridSearchCV(pipe, param_grid, scoring=scorer_r_p, cv=sss, n_jobs=-1)
	clf_grid.fit(features, labels)
	clf = clf_grid.best_estimator_
	return clf


def main():
	data_dict = dateset_summary()								# Load the dictionary containing the dataset
	outliers_names = ['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E']
	data_dict = remove_outerliers(data_dict,outliers_names) 	# remove outliers
	data_df = pd.DataFrame.from_dict(data_dict,orient='index')  # change dataset to dateframe format
	data_df['poi'] = data_df['poi'].astype(int)
	data_df = add_new_features(data_df) 						# create new features

	my_dataset, features_list = full_dataset_fatures(data_df)   # Store to my_dataset, features_list
	features, labels = get_features_labels(my_dataset,features_list)

	clf=train_classifier(features,labels)						#build classifier and tune parameters
	dump_classifier_and_data(clf, my_dataset, features_list)    # Dump classifier, dataset, and features_list

if __name__ == "__main__":
    main()
