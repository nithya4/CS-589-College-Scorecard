import csv
import io
import os
import glob
import numpy as np
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
Dictionary used to map filenames of years to categorical values
'''
fnameyear = {'MERGED1996_97_PP_pruned': 0,
             'MERGED1997_98_PP_pruned': 1,
             'MERGED1998_99_PP_pruned': 2,
             'MERGED1999_00_PP_pruned': 3,
             'MERGED2000_01_PP_pruned': 4,
             'MERGED2001_02_PP_pruned': 5,
             'MERGED2002_03_PP_pruned': 6,
             'MERGED2003_04_PP_pruned': 7,
             'MERGED2004_05_PP_pruned': 8,
             'MERGED2005_06_PP_pruned': 9,
             'MERGED2006_07_PP_pruned': 10,
             'MERGED2007_08_PP_pruned': 11,
             'MERGED2008_09_PP_pruned': 12,
             'MERGED2009_10_PP_pruned': 13,
             'MERGED2010_11_PP_pruned': 14,
             'MERGED2011_12_PP_pruned': 15,
             'MERGED2012_13_PP_pruned': 16,
             'MERGED2013_14_PP_pruned': 17,
             'MERGED2014_15_PP_pruned': 18}
# Mapping the categorical values back to File names.
filemap = {
    '0': 'MERGED1996_97_PP_pruned',
    '1': 'MERGED1997_98_PP_pruned',
    '2': 'MERGED1998_99_PP_pruned',
    '3': 'MERGED1999_00_PP_pruned',
    '4': 'MERGED2000_01_PP_pruned',
    '5': 'MERGED2001_02_PP_pruned',
    '6': 'MERGED2002_03_PP_pruned',
    '7': 'MERGED2003_04_PP_pruned',
    '8': 'MERGED2004_05_PP_pruned',
    '9': 'MERGED2005_06_PP_pruned',
    '10': 'MERGED2006_07_PP_pruned',
    '11': 'MERGED2007_08_PP_pruned',
    '12': 'MERGED2008_09_PP_pruned',
    '13': 'MERGED2009_10_PP_pruned',
    '14': 'MERGED2010_11_PP_pruned',
    '15': 'MERGED2011_12_PP_pruned',
    '16': 'MERGED2012_13_PP_pruned',
    '17': 'MERGED2013_14_PP_pruned',
    '18': 'MERGED2014_15_PP_pruned'
}

'''
Method to check if a directory exists. Create the directory if it does not exist.
'''


def check_directory(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


'''
Utility method to read the list of features used for prediction.
'''


def setup():
    # Path to files which contain the pruned list of features for training and list of features to predict.
    chosen_features = '/../HelperFiles/selected_features'
    predicted_features = '/../HelperFiles/predict_features'
    with open(os.path.dirname(__file__) + chosen_features) as f:
        f_list = f.readlines()
    f_list = [x.strip() for x in f_list]
    with open(os.path.dirname(__file__) + predicted_features) as f:
        p_list = f.readlines()
    p_list = [x.strip() for x in p_list]
    return f_list, p_list


'''
Utility method to map the variable names to their type
'''


def variable_map():
    fieldtype = {}
    fieldtype = defaultdict(lambda: None, fieldtype)
    # file which contains the variable , type(Continuous or categorical) map.
    file_name = '/../HelperFiles/feature_type.csv'
    with io.open(os.path.dirname(__file__) + file_name, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
        for row in reader:
            fieldtype[row['VARIABLE NAME']] = row['VARIABLE TYPE']

    return fieldtype


'''
Utility method to map variable type to variable name
'''


def get_variable_type():
    feature_file = '/../HelperFiles/feature_type.csv'
    variable_name_type_map = {"Categorical": [], "Continuous": []}
    with io.open(os.path.dirname(__file__) + feature_file, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
        for row in reader:
            variable_name_type_map[row["VARIABLE TYPE"]].append(row["VARIABLE NAME"])
    return variable_name_type_map


'''
Method to generate feature files
'''


def generate_feature_file(directory):
    f_list, p_list = setup()

    fieldnames = [x for x in f_list]
    for vals in p_list:
        fieldnames.append(vals)

    complete_data = {}
    complete_data = defaultdict(lambda: {}, complete_data)
    newfields = ['YEAR', 'UNITID', 'VALUE']
    os.chdir(directory)
    for file_name in glob.glob('*.csv'):
        with io.open(file_name, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
            key = file_name.split('.')
            k = key[0]
            year = fnameyear[k]
            for row in reader:
                for field in fieldnames:
                    if row[field].lower() != 'null' and row[field].lower() != 'privacysuppressed':
                        if field not in complete_data:
                            complete_data[field] = {}
                            complete_data[field][year] = {}
                        else:
                            if year not in complete_data[field]:
                                complete_data[field][year] = {}
                        complete_data[field][year][row['UNITID']] = row[field]
    feature_directory = directory + "FeatureData/"
    check_directory(feature_directory)
    for key, value in complete_data.items():
        write_file = feature_directory + key + '.csv'
        with open(write_file, 'w') as newfile:
            writer = csv.DictWriter(newfile, fieldnames=newfields)
            writer.writeheader()

            for year, vals in value.items():
                for college, val in vals.items():
                    newrow = {'YEAR': year, 'UNITID': college, 'VALUE': val}
                    writer.writerow(newrow)
    return feature_directory


'''
Method to stitch back data from across 10k+ colleges to yearwise csv files.
'''


def generate_year_wise_data(directory, output_directory):
    complete_data = {}
    complete_data = defaultdict(lambda: {}, complete_data)

    f_list, p_list = setup()
    fieldnames = [x for x in f_list]
    for vals in p_list:
        fieldnames.append(vals)
    fieldnames.insert(0, 'UNITID')

    os.chdir(directory)
    for file_name in glob.glob('*.csv'):
        with io.open(file_name, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
            for row in reader:
                temp = complete_data[row['YEAR']]
                if not row['UNITID'] in temp:
                    temp[row['UNITID']] = {}
                newdict = {}
                for key, value in row.items():
                    if key != 'YEAR':
                        newdict[key] = value
                temp[row['UNITID']] = newdict
                complete_data[row['YEAR']] = temp

    for year, values in complete_data.items():
        write_file = output_directory + filemap[year] + '.csv'
        with open(write_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for id_val, rows in values.items():
                writer.writerow(rows)

    return output_directory


# Create values and labels for line graphs
def plot_line(values, years, x_label, y_label, title):
    years = [i.split('.')[0].split('_')[0] for i in years]
    years = [i.split('MERGED')[-1] for i in years]
    years = [i[-2:] for i in years]
    values = np.array(values)  # the validation and training errors
    inds = range(len(years))
    # Plot a line graph
    plt.figure(2, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
    plt.plot(inds, values, 'sb-', linewidth=3)  # Plot the first series in blue with square marker

    # This plots the data
    plt.grid(True)  # Turn the grid on
    plt.ylabel(y_label)  # Y-axis label
    plt.xlabel(x_label)  # X-axis label
    plt.title(title)  # Plot title
    plt.xticks(inds, years)

    plt.tight_layout()

    # Save the chart
    plt.savefig(os.path.dirname(__file__) + "/../Figures/" + title + ".png")
    plt.close()
