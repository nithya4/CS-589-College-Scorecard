import csv
import io
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
Method to plot the regession curve and find average values for a particular feature for a particular college.
'''


def regression(X, Y, known_year, value, unitid, list_years):
    params = {}
    clf = linear_model.Ridge(**params)
    clf.fit(X, Y)
    weights = clf.coef_
    C = clf.intercept_

    pred_value = weights[known_year] * 1 + C
    diff = pred_value - value
    C_hat = C - diff
    value_list = []

    for year in list_years:
        prediction = max(weights[year[0]] * 1 + C_hat, 0)
        value_list.append(prediction)

    return value_list


'''
Convert categorical variables to binary vectors.
'''


def encode_data(data, known_year, value, unitid, list_years):
    total_data = np.asarray(data).astype(np.float)
    X = total_data[:, 1:total_data.shape[1] - 1]  # total_data[0] is simply UNITID - not needed for regression
    Y = total_data[:, -1]

    enc = OneHotEncoder(n_values=[19], categorical_features=[0], sparse=False)
    X_hat = enc.fit_transform(X)

    return regression(X_hat, Y, known_year, value, unitid, list_years)


'''
Main method which is called to plot a trend line and obtain missing values.
'''


def get_value_list(feature_name, unitid, value, known_year, list_years, feature_directory):
    data = []
    with io.open(feature_directory + feature_name + '.csv', 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
        for row in reader:
            data.append([row['UNITID'], row['YEAR'], row['VALUE']])

    return encode_data(data, known_year, value, unitid, list_years)
