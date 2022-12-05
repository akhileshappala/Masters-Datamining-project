import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt

###################
# RUN EVALUATION

def run_all_regressors(X_train, y_train, X_test, y_test):
    regressor_list = {
        "RandomForestRegressor" : RandomForestRegressor(),
        "GradientBoostingRegressor" : GradientBoostingRegressor(),
        "DecisionTreeRegressor" : DecisionTreeRegressor(),
        "ExtraTreesRegressor": ExtraTreesRegressor(),
        "SVR": SVR()
    }
    regressorDict={}
    for type in regressor_list.keys():
        print(f"Running {type}")
        resultList = run_regressor(X_train, y_train, X_test, y_test, regressor_list[type])
        regressorDict[type] = resultList
    return regressorDict

def run_regressor(X_train, y_train, X_test, y_test, regressor_type):
    model = regressor_type
    model.fit(X_train, y_train)
    
    trainScore= model.score(X_train, y_train)
    testScore = model.score(X_test, y_test)
    print(f'Score on training data: {trainScore}')
    print(f'Score on testing data: {testScore}')

    predictions = model.predict(X_test)

    mean_absolute_error_value = mean_absolute_error(y_test, predictions)
    print('Mean Absolute Error: ', mean_absolute_error_value)
    r2_score_value = r2_score(y_test, predictions)
    print('R Squared: ', r2_score_value)
    #adj_R2_value = adj_R2(r2_score_value, X_test.size, len(X_test.columns))
    #print('Adjusted R Squared: ', adj_R2_value)

    print()
    return [trainScore,testScore,mean_absolute_error_value,r2_score_value]

def adj_R2(R2, n, p):
    r2 = 1-(1-R2)*(n-1)/(n-p-1)
    return r2


def run_all_classifiers(X_train, X_test, y_train, y_test):
    classifier_list = {
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "ExtraTreesClassifier": ExtraTreesClassifier(),
        "SVC": SVC()
    }
    classifier_dict={}
    for type in classifier_list.keys():
        print(f"Running {type}")
        result = run_classifier(X_train, X_test, y_train, y_test, classifier_list[type])
        classifier_dict[type] = result
    return classifier_dict

def run_classifier(X_train, X_test, y_train, y_test, classifier_type):
    model = classifier_type
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(predictions[:5])
    print(y_test.iloc[:5])
    acc_score = accuracy_score(y_test, predictions)
    #prec_score = precision_score(y_test, predictions, average="weighted")
    sensitivity = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")
    print("Accuracy: ", acc_score)
    print("Sensitivity: ", sensitivity)
    print("F1 Score: ", f1)
    return [acc_score,sensitivity,f1]

#########################

########################
# HELPER

def reduce_cause_labels(df):
    reduced_labels = ['Children', 'Smoking', 'Campfire', 'Railroad', 'Structure', 'Powerline', 'Fireworks']
    df = df.loc[df['stat_cause_descr'] != 'Missing/Undefined']
    df['stat_cause_descr'] = df['stat_cause_descr'].apply(lambda x: 'Other' if (x in reduced_labels) else x)
    return df
