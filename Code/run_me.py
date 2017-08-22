import glob
import os
import imputations
import utils
import preprocessing
import regression
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Path to the directory containing the data.
DATA_DIR = "/Users/rakshitha/CSCI-589-Project_final/CollegeScorecard_Raw_Data/"

# Performing initial setup
f_list, p_list = utils.setup()
# Preprocessing Stage.

# 1. Pruning the original csv files

os.chdir(DATA_DIR)
for file_name in glob.glob('*.csv'):
    fnames = file_name.split('.')
    fname = fnames[0]
    output_dir = preprocessing.process_file(fname, f_list, p_list, DATA_DIR)
# # 2. Generate collegewise data
processed_dir = preprocessing.generate_college_csv(f_list, p_list, output_dir)
print " Preprocessing done"
# Imputation Stage

# 1. Fill missing values for each feature for each college.
# 1a. We need to generate feature.csv files which assist in imputation value computations.
feature_directory = utils.generate_feature_file(output_dir)
print "Feature csv files created"
# call the collegewise imputation function.
filled_directory = imputations.impute_college_data(f_list, p_list, processed_dir, feature_directory)
# Our next step is to transform this collegewise information to the original yearwise format in
# order to perform regression.
stitched_directory = utils.generate_year_wise_data(filled_directory, output_dir)
print " Stitching Done"
# There were a few features which were missing across all years for some colleges. hence when the data
#  was transformed it still had missing values for certain datapoints for certain features. In order to fill in
# these missing values, we again ran mean, median, mode imputations. Some features had null values for
#  all data points in a year. These features were discarded.
cleaned_filled_directory = imputations.impute_year_data(f_list, p_list, stitched_directory)
print " Cleaning done"
# The last phase is the regression phase. Here ew first convert categorical data into binary categorical data using
# OneHotEncoder. We then perform dimensionality reduction using PCA. The transformed feature vectors are then passed
# to the regressors to perform prediction
regression.run_regressor(cleaned_filled_directory)
print "Regression done"
