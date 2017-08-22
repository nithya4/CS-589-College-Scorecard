import csv
import io
import glob
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from regression_pipeline import caller
from sklearn.ensemble import RandomForestRegressor
import utils

f_list, p_list = utils.setup()

'''
Fixes the value for categorical variables for whom the range is not [0,num of categories)
'''


def fix_feature(value, name, range_val):
    value = int(float(value))
    if name in ['CONTROL', 'LOCALE2', 'ICLEVEL']:
        return value - 1
    elif name in ['CCUGPROF', 'CCSIZSET']:
        if value == -2:
            return int(float(range_val)) - 1
        else:
            return value
    else:
        value = str(value)
        localemap = {'11': '0', '12': '1', '13': '2', '21': '3', '22': '4', '23': '5', '31': '6', '32': '7', '33': '8',
                     '41': '9', '42': '10', '43': '11', '-3': '12'}
        return int(float(localemap[value]))


'''
Pipeline which first converts categorical variables to binary categorical values using a one hot encoder.
It next reduces the feature vector using PCA(dimensionality reduction).
'''


def run_regressor(directory):
    feature_file_path = '/../HelperFiles/feature_type.csv'
    feature_map = {}
    feature_map = defaultdict(lambda: {}, feature_map)
    # Add comment as to what it is doing here.
    with io.open(os.path.dirname(__file__) + feature_file_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
        for row in reader:
            feature_map[row['VARIABLE NAME']] = row

    os.chdir(directory)
    result_set = {"pca_n_components": {"year": [], "vals": []}, "RMSE_debt": {"year": [], "vals": []},
                  "RMSE_multivariate": {"year": [], "vals": []}, "RMSE_earnings": {"year": [], "vals": []},
                  "mean_debt": {"year": [], "vals": []}, "mean_earnings": {"year": [], "vals": []}}
    for file_path in glob.glob('*.csv'):
        # performs encoding, pca reduction and regression for each year.
        X_data = []
        Y_data = []
        check_predict = True
        with io.open(file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
            for x in p_list:
                if x not in reader.fieldnames:
                    check_predict = False
                    break
            if not check_predict:
                continue

            for row in reader:
                count = 0
                for feature_name, feature_value in row.items():
                    if feature_name not in p_list + ['UNITID']:
                        feature_info = feature_map[feature_name]
                        if feature_info['FIT FOR ENCODING'] == 'FALSE':
                            # If the feature requires encoding, then perform encoding. Otherwise ignore.
                            new_value = fix_feature(feature_value, feature_name, feature_info['NUMBER OF CATEGORIES'])
                            row[feature_name] = new_value
                            count += 1
                # UNITID is not a feature for prediction. Hence deleting it.
                del row['UNITID']
                temp_list = []
                target_list = []
                for field in reader.fieldnames:
                    if field not in p_list + ['UNITID']:
                        temp_list.append(row[field])
                for predict_field in p_list:
                    target_list.append(row[predict_field])
                X_data.append(temp_list)
                Y_data.append(target_list)
        # This loop computes total categorical features that needs to be changed by the OneHot Encoder.
        num_values = []
        categorical_features = []
        count = 0
        for field in reader.fieldnames:
            if field not in p_list + ['UNITID']:
                feature_info = feature_map[field]
                if feature_info['VARIABLE TYPE'] != 'Continuous':
                    categorical_features.append(count)
                    num_values.append(feature_info['NUMBER OF CATEGORIES'])
                count += 1

        X = np.asarray(X_data, dtype=float)
        Y = np.asarray(Y_data, dtype=float)

        enc = OneHotEncoder(n_values=num_values, categorical_features=categorical_features, sparse=False)
        enc.fit(X)
        X_hat = enc.transform(X)

        pca = PCA(n_components=0.99, random_state=0, whiten=True)
        pca.fit(X_hat)
        X_transformed = pca.transform(X_hat)
        print "*********************", file_path, "**********************"
        print pca.n_components_
        result_set["pca_n_components"]["year"].append(file_path)
        result_set["pca_n_components"]["vals"].append(pca.n_components_)
        # print pca.components_
        baseline_model = Ridge()
        baseline_params = {"alpha": [0.1, 1, 10, 100, 1000]}
        K = 5
        # Execution for baseline model.
        print "Baseline Debt"
        caller(X_transformed, Y[:, 0], baseline_model, baseline_params, K)
        print "Baseline Earnings"
        caller(X_transformed, Y[:, 1], baseline_model, baseline_params, K)
        # Multiple output
        print "Baseline Both"
        caller(X_transformed, Y, baseline_model, baseline_params, K)

        model = RandomForestRegressor(random_state=0)
        params = {"max_depth": [10, 30, 50], "n_estimators": [10, 30, 50]}

        print "RF Debt"
        debt_rmse, debt_mean = caller(X_transformed, Y[:, 0], model, params, K)
        result_set["RMSE_debt"]["year"].append(file_path)
        result_set["RMSE_debt"]["vals"].append(debt_rmse)
        result_set["mean_debt"]["year"].append(file_path)
        result_set["mean_debt"]["vals"].append(debt_mean)
        print "RF Earnings"
        earnings_rmse, earnings_mean = caller(X_transformed, Y[:, 1], model, params, K)
        result_set["RMSE_earnings"]["year"].append(file_path)
        result_set["RMSE_earnings"]["vals"].append(earnings_rmse)
        result_set["mean_earnings"]["year"].append(file_path)
        result_set["mean_earnings"]["vals"].append(earnings_mean)
        # Multiple output
        print "RF Both"
        both_rmse, mean_both = caller(X_transformed, Y, model, params, K)
        result_set
        result_set["RMSE_multivariate"]["year"].append(file_path)
        result_set["RMSE_multivariate"]["vals"].append(both_rmse)

    for plot_title in result_set:
        utils.plot_line(result_set[plot_title]["vals"], result_set[plot_title]["year"], "Year", plot_title, "YearVs"+plot_title)
