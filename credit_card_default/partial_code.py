# -*- coding: utf-8 -*-
"""
In this assessment you will need do perform a simple machine learning
task using sklearn. Partial code are provided. Please fill in
the missing code as per instructed
"""
import numpy as np
from sklearn.svm import SVC
import pandas as pd
import csv
import sklearn
import os
import pprint

pp = pprint.PrettyPrinter(indent = 1)
"""
Please assign path of DefaultRecord_Person.csv to variable fileA
Please assign path of DefaultRecord_History.csv to variable fileB
"""
fileA = os.path.join("/home/victor/workspace/machine-learning/credit_card_default","DefaultRecord_Person.csv")
fileB = os.path.join("/home/victor/workspace/machine-learning/credit_card_default","DefaultRecord_History.csv")
#Insert your code here
print fileA
print fileB
"""
Please write a simple function named 'readdata'
that take in the file path of the source data and return
data read from csv file.
Please use the csv model instead of the pandas read_csv function
"""
#Insert your code here
def readdata(path):
    with open(path, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        rows = []
        for row in reader:
            rows.append(row)
    return rows
"""
read data from fileA and fileB respectively
using readdata function
"""

dataA, dataB = readdata(fileA), readdata(fileB)


"""
check number of row in dataA & dataB
please print the number of row in dataA and dataB
"""
print len(dataA)
print len(dataB)
#Insert your code here


"""
remove the first row of both dataA & dataB
and assign the first row of dataA and dataB to
dataA_header and dataB_header respectively
"""
dataA_header = dataA[0]
dataB_header = dataB[0]
dataA.pop(0)
dataB.pop(0)

#Insert your code here


"""
print the first 5 rows in dataA
"""
#Insert your code here
pp.pprint(dataA[0:5])


"""
Please load dataA and dataB into pandas Dataframe as df_A
and df_B, and with column of dataA_header & dataB_header
respectively.
"""
#Insert your code here
df_A = pd.read_csv(fileA)
df_B = pd.read_csv(fileB)

"""
join both dataframe using column ID and assign the joined table
to df_data. You may want to set ID column as index for
both Dataframe.
"""
df_A = df_A.set_index('ID')
df_B = df_B.set_index('ID')
df_data = df_A.join(df_B)


"""
Please prepare your data for machine learning. Please be ready to explain
the rational of how you prepare your data.
"""

features = pd.DataFrame()
features = df_data.drop(['default payment next month'],axis = 1)
response = pd.DataFrame()
response['default payment next month'] = df_data.loc[:,'default payment next month']

print features.corr(method="pearson", min_periods=1)
"""
Please split your data into training set and testing set using train_test_split
provided by sklearn.

For example if your using 'LIMIT_BAL', 'SEX', 'EDUCATION' & 'MARRIAGE'
to predict 'default payment next month', you can try

X_train, X_test, y_train, y_test = train_test_split(df_data[['MARRIAGE','SEX','EDUCATION']], response, test_size = 0.2, random_state = 42)

"""

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, response, test_size = 0.2, random_state = 42)
#insert your code here
"""
Feature selection Selecting K Best Features
"""

selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, k=5)
X_selected = selector.fit_transform(X_train, y_train.values.ravel())
"""
selector = SelectKBest(mutual_info_classif, k=5)
X_selected = selector.fit_transform(X_train, y_train.values.ravel(),
discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
"""

selected_columns = []
selected_flags = selector.get_support()
for i in range(0, len(selected_flags) - 1):
    if(selected_flags[i] == True):
        selected_columns.append(X_train.columns[i])
print "The following features are selected: " + str(selected_columns)

"""
Please train your classifer using model of your choice.
"""

#insert your code here
model = sklearn.linear_model.LogisticRegression(penalty = 'l2', C = 1)
model.fit(X_selected, y_train.values.ravel())
#Cs = np.logspace(-6, -1, 10)
#lf = GridSearchCV(estimator= SVC(kernel = 'linear', C = 1), param_grid=dict(C=Cs), n_jobs=-1)
#clf.fit(X_selected, y_train.values.ravel())
#scores = clf.best_score_
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


clf = sklearn.svm.SVC(kernel = 'linear', C = 1)
# scores = cross_val_score(clf, features[selected_columns], response.values.ravel(), cv=5,n_jobs = -1)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf.fit(X_selected,y_train.values.ravel())


"""
Please print our classification report using classification_report provided
by sklearn to evalulate your classifer.
"""

#insert your code here
print "Logistic accuracy is %2.2f" % sklearn.metrics.accuracy_score(y_test, model.predict(X_test[selected_columns]))
print "SVM accuracy is %2.2f" % sklearn.metrics.accuracy_score(y_test, clf.predict(X_test[selected_columns]))
print clf.get_params()
