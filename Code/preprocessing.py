import csv
import glob
import io
import os

from collections import defaultdict

import utils

'''
Method to prune the original CSV files with all features to csv files with chosen features and prediction features.
'''


def process_file(fname, f_list, p_list, directory):
    modified_data = []
    fieldnames = [x for x in f_list]
    for vals in p_list:
        fieldnames.append(vals)
    fieldnames.insert(0, 'UNITID')

    with io.open(fname + '.csv', 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
        for row in reader:
            modified_row = {}
            for field in fieldnames:
                modified_row[field] = row[field]
            modified_data.append(modified_row)

    # The processed files are written to the output directory.
    output_dir = directory + 'Processed/'
    utils.check_directory(output_dir)

    write_file = output_dir + fname + '_pruned.csv'
    with open(write_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in modified_data:
            writer.writerow(row)

    return output_dir


'''
Method to convert the yearwise csv files into collegewise csv files.
This is done to perform imputations on collegewise data.
'''


def generate_college_csv(f_list, p_list, directory):
    fieldnames = [x for x in f_list]
    for vals in p_list:
        fieldnames.append(vals)
    fieldnames.insert(0, 'YEAR')
    fieldnames.insert(0, 'UNITID')

    complete_data = {}
    complete_data = defaultdict(lambda: {}, complete_data)

    # In order to impute missing values for a particular college across years, we need to map
    # filenames to corresponding categorical values.
    fnameyear = utils.fnameyear
    # Changing the directory to the path containing the pruned_csv_files.
    os.chdir(directory)
    # reading information from all the 19 years and storing collegewise data into 'complete_data' dictionary.
    for file_name in glob.glob('*.csv'):
        with io.open(file_name, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader((l.encode('utf-8') for l in csvfile))
            key = file_name.split('.')
            k = key[0]
            # Mapping the filename to the corresponding categorical value.
            year = fnameyear[k]
            for row in reader:
                row['YEAR'] = year
                temp = complete_data[row['UNITID']]
                temp[year] = row
                complete_data[row['UNITID']] = temp

    # Writing back the college information into file. These files will be used for further processing.
    output_dir = directory + 'CollegeData/'
    utils.check_directory(output_dir)
    for key, value in complete_data.items():
        write_file = output_dir + key + '.csv'
        with open(write_file, 'w') as newfile:
            writer = csv.DictWriter(newfile, fieldnames=fieldnames)
            writer.writeheader()
            for year, vals in value.items():
                writer.writerow(vals)

    return output_dir
