import csv
import glob
import io
import math
import os
import warnings
from collections import defaultdict
from random import shuffle

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder

import utils
from trends import get_value_list

warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
Method to compute missing values from continuous valued features using a Regression line.
'''


def generateLinearRegressionLine(train_x, train_y, test_x, test_y=None):
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    if test_y:
        test_y = np.asarray(test_y, dtype=float)
    reg = Ridge()
    reg.fit(train_x, train_y)
    pred_y = reg.predict(test_x)
    if test_y is not None:
        print test_y
        print mean_squared_error(test_y, pred_y)
        print "*" * 10
        print
    return list(pred_y)


'''
Method to perform imputations on missing values.
'''


def impute_college_data(f_list, p_list, directory, feature_directory):
    fieldnames = [x for x in f_list]
    for vals in p_list:
        fieldnames.append(vals)
    fieldnames.insert(0, 'UNITID')
    fieldtype = utils.variable_map()
    os.chdir(directory)
    # for each of the college files, perform imputations.
    for file_name in glob.glob('*.csv'):
        fieldValues = {}  # this dictionary is used to store the set of rows which could be used as training data for a feature.
        fieldstats = {}
        fieldstats = defaultdict(lambda: 0, fieldstats)
        unitid = file_name.split('.csv')[0]
        csv_map = {}  # this dictionary is used to store the imputed values per college.
        # initialize the fieldnames dictionary. The test_x and test_y rows were used to
        # obtain the RMSE scores and validate the choice of regression method chosen.
        for field in fieldnames:
            fieldValues[field] = {"train_x": [], "train_y": [], "test_x": [], "test_y": [], "main_predict_x": [],
                                  "main_predict_y": []}

        with io.open(file_name, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
            count = 0
            for row in reader:
                count += 1
                for var_name in fieldnames:
                    if row[var_name].lower() != 'null' and row[var_name].lower() != 'privacysuppressed':
                        fieldstats[var_name] += 1
                        fieldValues[var_name]["train_x"].append([row["YEAR"]])
                        fieldValues[var_name]["train_y"].append(row[var_name])
                    else:
                        # These are the set of rows which need to be imputed.
                        fieldValues[var_name]["main_predict_x"].append([row["YEAR"]])
                        fieldValues[var_name]["main_predict_y"].append(np.NaN)
            fieldstats['total_rows'] = count

            # Depending on the type of variable, we call either mode imputation or linear regressor
            # to compute the missing values.
            for var_name in fieldtype.keys():
                combined_X = fieldValues[var_name]["train_x"] + fieldValues[var_name]["main_predict_x"]
                combined_Y = fieldValues[var_name]["train_y"] + fieldValues[var_name]["main_predict_y"]
                fieldValues[var_name]["combined_x"] = combined_X
                fieldValues[var_name]["combined_y"] = combined_Y

                if fieldtype[var_name] == 'Categorical':

                    if 1 <= fieldstats[var_name] < fieldstats['total_rows']:
                        # performing mode imputation.
                        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
                        imputed_val = imp.fit_transform(combined_Y)
                        fieldValues[var_name]["combined_y"] = imputed_val[0]
                else:
                    # Imputations for continuous valued variable.
                    # In case more than two points were present for a particular feature per year for a
                    #  college, we would plot a regression line of the values of the features versus year. Since year
                    # was a categorical data, we converted the year into binary categorical using one hot encoder
                    # and then performed the imputations.

                    if 2 <= fieldstats[var_name] < fieldstats['total_rows']:
                        enc = OneHotEncoder(n_values=[19], categorical_features=[0], sparse=False)
                        '''
                            Code to test the RMSE scores of the method. We found considerably low RMSE score
                            for Ridge over Lasso or OLS method. Hence Ridge was chosen.
                        '''
                        # split_ratio = int(len(fieldValues[var_name]["train_x"]) * 0.7)
                        # train_split = fieldValues[var_name]["train_x"][:split_ratio]
                        # train_label_split = fieldValues[var_name]["train_y"][:split_ratio]
                        # test_split = fieldValues[var_name]["train_x"][split_ratio:]
                        # test_label_split = fieldValues[var_name]["train_y"][split_ratio:]
                        '''
                            Trnasforming into One hot encoding format.
                        '''
                        X_transformed_features = enc.fit_transform(fieldValues[var_name]["train_x"])
                        test_X_transformed = enc.transform(fieldValues[var_name]["main_predict_x"])
                        # catch_var = generateLinearRegressionLine(X_transformed_features, train_label_split,
                        # test_X_transformed, test_label_split)
                        '''
                            You get back the predicted values for the missing variables.
                        '''
                        fieldValues[var_name]["main_predict_y"] = generateLinearRegressionLine(X_transformed_features,
                                                                                               fieldValues[var_name][
                                                                                                   "train_y"],
                                                                                               test_X_transformed)
                        # Combining the dataset.
                        combined_Y = fieldValues[var_name]["train_y"] + fieldValues[var_name]["main_predict_y"]
                        fieldValues[var_name]["combined_y"] = combined_Y
                    # In case we have only one value for a particular feature, we average based off of trends
                    # for this feature over the years across all colleges.

                    elif fieldstats[var_name] == 1 and fieldstats['total_rows'] != 1:
                        fieldValues[var_name]["main_predict_y"] = get_value_list(var_name, int(unitid), float(
                            fieldValues[var_name]["train_y"][0]), int(fieldValues[var_name]["train_x"][0][0]),
                                                                                 np.asarray(fieldValues[var_name][
                                                                                                "main_predict_x"],
                                                                                            dtype=int),
                                                                                 feature_directory)
                        combined_Y = fieldValues[var_name]["train_y"] + fieldValues[var_name]["main_predict_y"]
                        fieldValues[var_name]["combined_y"] = combined_Y
                # Mapping phase. We have now filled up the missing values for the feature.
                # We store this information in the csv_map data structure to be written to the file later.
                for index in range(len(fieldValues[var_name]["combined_x"])):
                    year = fieldValues[var_name]["combined_x"][index][0]
                    if year not in csv_map:
                        csv_map[year] = {}
                    csv_map[year][var_name] = fieldValues[var_name]["combined_y"][index]
                    csv_map[year]["YEAR"] = year
                    csv_map[year]["UNITID"] = unitid

        # We now write back the missing values into the file.
        write_file = directory + file_name
        with open(write_file, 'w') as newfile:
            writer = csv.DictWriter(newfile, fieldnames=csv_map.values()[0].keys())
            writer.writeheader()
            writer.writerows(csv_map.values())
    # return the directory where the files are written
    return directory


'''
Method to perform imputations using sklearn's imputer function.
Returns the RMSE score.
'''


def impute(training_x, y_true, strategy):
    imp = Imputer(missing_values='NaN', strategy=strategy, axis=1)
    imputed_val = imp.fit_transform(training_x)
    y_true = np.asarray(y_true, dtype=float)
    mse = mean_squared_error(y_true, imputed_val[0])
    return np.sqrt(mse)


'''
Method which will call the imputer and fill the missing data.
'''


def fill_data(input_data, strategy, feature):
    impute_input = []
    missing_values = ["null", "privacysuppressed", "nan", ""]
    for row in input_data:
        if row[feature].lower() in missing_values:
            impute_input.append(np.NAN)
        else:
            impute_input.append(row[feature])
    imp = Imputer(missing_values='NaN', strategy=strategy, axis=1)
    imputed_values = imp.fit_transform(impute_input)[0]
    for i in range(len(input_data)):
        input_data[i][feature] = imputed_values[i]
    return input_data


'''
Performs train test split to identify the imputation strategy which will give the least RMSE scores.
'''


def compute_imputations(data, feature, var_type):
    # rows without null values
    available_count = len(data["data_available"])
    if available_count == 0:
        # This means all values are missing.
        return data["combined"]
    # in case there are no null rows, there's no need to perform imputations. return it.
    elif len(data["null_rows"]) == 0:
        return data["data_available"]
    # performing the train test split to identify the best imputation strategy for a feature.
    shuffle(data["data_available"])
    row_count = len(data["data_available"])
    split_ratio = int(math.ceil(row_count * 0.7))
    training = data["data_available"][:split_ratio]
    test = data["data_available"][split_ratio:]
    training_x = []
    y_true = []
    for row in test:
        training_x.append(np.NaN)
        y_true.append(row[feature])
    for row in training:
        training_x.append(row[feature])
        y_true.append(row[feature])
    training_x = np.asarray(training_x, dtype=float)
    # test_x = np.array(test_x)
    # check the type of the variable. if it is categorical, perform median and mode imputation
    # and choose one which gives least rmse. Similaraly for continuous , perform mean, mode and median and
    #  report the method which performs the best.
    if var_type == "Categorical":
        rmse_mode = impute(training_x, y_true, "most_frequent")
        rmse_median = impute(training_x, y_true, "median")
        if rmse_mode < rmse_median:
            strategy = "most_frequent"
        else:
            strategy = "median"
        data["combined"] = fill_data(data["data_available"] + data["null_rows"], strategy, feature)
        return data["combined"]
    # variable is continuous. So we check against mean, median, mode
    else:
        rmse_mode = impute(training_x, y_true, "most_frequent")
        rmse_median = impute(training_x, y_true, "median")
        rmse_mean = impute(training_x, y_true, "mean")
        if rmse_mode < rmse_median and rmse_mode < rmse_mean:
            strategy = "most_frequent"
        elif rmse_median < rmse_mean:
            strategy = "median"
        else:
            strategy = "mean"
        data["combined"] = fill_data(data["data_available"] + data["null_rows"], strategy, feature)
        return data["combined"]


'''
This method reads in all the variables of a csv file, categorizes the rows
into value present or absent per feature and performs imputations.

'''


def categorize_rows(fname, f_list, p_list, dir):
    fieldnames = [x for x in f_list]
    data_repo = {}
    yearwise_info = {}
    missing_values = ["null", "privacysuppressed", "nan", ""]
    for vals in p_list:
        fieldnames.append(vals)
    for field in fieldnames:
        data_repo[field] = {"null_rows": list(), "data_available": list(), "combined": list()}
    with io.open(fname + '.csv', 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
        for row in reader:
            yearwise_info[row["UNITID"]] = row
            for field in fieldnames:
                if row[field].lower() in missing_values:
                    data_repo[field]["null_rows"].append(row)
                else:
                    data_repo[field]["data_available"].append(row)
    variable_map = utils.get_variable_type()
    for field in fieldnames:
        if field in variable_map["Continuous"]:
            var_type = "Continuous"
        else:
            var_type = "Categorical"
        data_repo[field]["combined"] = compute_imputations(data_repo[field], field, var_type)
        # in case all the data points have null values for the feature, discard it.
        if len(data_repo[field]["combined"]) == 0:
            for unitid in yearwise_info:
                del yearwise_info[unitid][field]
        else:
            for row in data_repo[field]["combined"]:
                yearwise_info[row["UNITID"]][field] = row[field]
    return yearwise_info


'''
Main method which is called to impute missing values across years.
'''


def impute_year_data(f_list, p_list, directory):
    os.chdir(directory)
    for file_name in glob.glob('*.csv'):
        fnames = file_name.split('.')
        fname = fnames[0]
        yearwise_info = categorize_rows(fname, f_list, p_list, directory)
        write_file = directory + fname + '.csv'
        with open(write_file, 'w') as newfile:
            yearwise_info.values()[0].keys()
            writer = csv.DictWriter(newfile, fieldnames=yearwise_info.values()[0].keys())
            writer.writeheader()
            writer.writerows(yearwise_info.values())
    return directory
