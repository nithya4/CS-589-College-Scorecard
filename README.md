# To College or Not to College: Prediction of Post Collegiate Earnings and Debts

## Machine Learning, Missing Data Completion, Average Trends, Single and Multivariate Regression

The Project consists of three folders: Code, Figures and HelperFiles. The Code folder consists of all the files required to run the program. The Figures folder consists of a list of images depicting the results and trends obtained. The HelperFiles folder consists of a list of files which are read by the programs to perform computation. These files provide a list of the features chosen, the list of prediction variables and the map of the variables with their type : Categorical or Continuous.

How to run the project: 
- The project can be executed by running the run_me.py file.
- The dataset can be downloaded from https://collegescorecard.ed.gov/data/documentation/
- The zip file needs to be extracted inside the MLProject folder along with the Code, Figures and HelperFiles directories. 
- The path to the file needs to be changed in the run_me.py file. 
- The run_me file first calls the functions from the preprocessing.py file to perform preprocessing. 
- The next step is to perform imputations on the missing data. 
The functions related to imputations are present in the imputations.py file. 
- We then run the regression.py file which contains methods to transform the data, perform dimensionality reduction and initalize a regressor. 
- The regression_pipeline.py file consists of methods to perform cross validation , choose the best hyperparameter and compute the RMSE scores. The utils file consists of miscellaneous utility methods.

# Tools and Technologies used:
- Python : 2.7
- Sklearn: 0.18 

# Trend Predictions:
Mean Earnings
![alt text](https://github.com/nithya4/CS-589-College-Scorecard/blob/master/Figures/YearVsmean_earnings.png "Average Earnings")

Mean Debt
![alt text](https://github.com/nithya4/CS-589-College-Scorecard/blob/master/Figures/YearVsmean_debt.png "Average Debt")
