# -*- coding: utf-8 -*-
import os
import csv
import pandas as pd
import pprint
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def readdata(path):
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = []
        for row in reader:
            rows.append(row)
    return rows


def main():
    pp = pprint.PrettyPrinter(indent=1)
    fileA = os.path.join(
        "/home/victor/workspace/machine-learning/credit_card_default",
        "DefaultRecord_Person.csv")
    fileB = os.path.join(
        "/home/victor/workspace/machine-learning/credit_card_default",
        "DefaultRecord_History.csv")
    print fileA
    print fileB
    dataA, dataB = readdata(fileA), readdata(fileB)
    print len(dataA)
    print len(dataB)
    dataA_header = dataA[0]
    dataB_header = dataB[0]
    dataA.pop(0)
    dataB.pop(0)
    pp.pprint(dataA[0:5])
    df_A = pd.read_csv(fileA)
    df_B = pd.read_csv(fileB)
    df_A = df_A.set_index('ID')
    df_B = df_B.set_index('ID')
    df_data = df_A.join(df_B)
    """
    Machine Learning preparation
    """

    features = pd.DataFrame()
    # Drop the response variable from the DataFrame
    features = df_data.drop(['default payment next month'], axis=1)

    # Here is the response variable
    response = pd.DataFrame()
    response['default payment next month'] = df_data.loc[
        :, 'default payment next month']

    print features.corr(method="pearson", min_periods=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, response, test_size=0.2, random_state=42)

    """
    Feature selection Selecting K Best Features
    """

    selector = SelectKBest(
        f_classif, k=5)
    X_selected = selector.fit_transform(X_train, y_train.values.ravel())
    """
    selector = SelectKBest(mutual_info_classif, k=5)
    X_selected = selector.fit_transform(X_train, y_train.values.ravel(),
    discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
    """

    selected_columns = []
    selected_flags = selector.get_support()
    for i in range(0, len(selected_flags) - 1):
        if(selected_flags[i]):
            selected_columns.append(X_train.columns[i])
    print "The following features are selected: " + str(selected_columns)

    """
    Please train your classifer using model of your choice.
    """

    # LogisticRegression model fitting
    model = LogisticRegression(penalty='l2', C=1)
    model.fit(X_selected, y_train.values.ravel())

    # SVM model fitting
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_selected, y_train.values.ravel())

    """
    Please print our classification report using classification_report provided
    by sklearn to evalulate your classifer.
    """
    print "Logistic accuracy is %2.2f" % accuracy_score(
        y_test, model.predict(X_test[selected_columns]))
    print "SVM accuracy is %2.2f" % accuracy_score(
        y_test, clf.predict(X_test[selected_columns]))
    print clf.get_params()
    return


main()
