import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt


def run_all_regressors(X_train, y_train, X_test, y_test):
    regressor_list = {
        "RandomForestRegressor" : RandomForestRegressor(),
        "GradientBoostingRegressor" : GradientBoostingRegressor(),
        "DecisionTreeRegressor" : DecisionTreeRegressor(),
        "ExtraTreesRegressor": ExtraTreesRegressor()

    }

    for type in regressor_list.keys():
        print(f"Running {type}")
        run_regressor(X_train, y_train, X_test, y_test, regressor_list[type])
    

def run_regressor(X_train, y_train, X_test, y_test, regressor_type):
    model = regressor_type
    model.fit(X_train, y_train)

    print(f'Score on training data: {model.score(X_train, y_train)}')
    print(f'Score on testing data: {model.score(X_test, y_test)}')

    predictions = model.predict(X_test)

    mean_absolute_error_value = mean_absolute_error(y_test, predictions)
    print('Mean Absolute Error:', mean_absolute_error_value)
    r2_score_value = r2_score(y_test, predictions)
    print('R Squared:', r2_score_value)
    adj_R2_value = adj_R2(r2_score_value, X_test.size, len(X_test.columns))
    print('Adjusted R Squared: ', adj_R2_value)

    print()

def adj_R2(R2, n, p):
    r2 = 1-(1-R2)*(n-1)/(n-p-1)
    return r2

def run_all_classifiers(X_train, y_train, X_test, y_test):
    classifier_list = {
        
    }