from math import sqrt

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

'''
Computes the RMSE score
'''


def RMSE(real_y, pred_y):
    rms = sqrt(mean_squared_error(real_y, pred_y))
    return rms


RMSE_Score = make_scorer(RMSE, False)
'''
Method to select the best hyperparameter values for the given model.
'''


def hyperparameter_optimization(model, params, X, Y, K):
    clf = GridSearchCV(model, params, cv=K, scoring=RMSE_Score)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    scores = RMSE(Y_test, Y_pred)
    if len(Y_pred.shape) == 2:
        mean_of_predictions = (np.mean(Y_pred[0]), np.mean(Y_pred[1]))
    else:
        mean_of_predictions = np.mean(Y_pred)
    return scores, clf.best_params_, clf.cv_results_['mean_train_score'], clf.cv_results_[
        'mean_test_score'], mean_of_predictions


def pipeline(X, Y, model, params, K):
    scores, best_params, train_scores, test_scores, mean_of_predictions = hyperparameter_optimization(model, params, X,
                                                                                                      Y, K)
    return best_params, train_scores, test_scores, scores, mean_of_predictions


def caller(X, Y, model, params, K):
    best_params, train_scores, test_scores, scores, mean_of_predictions = pipeline(X, Y, model, params, K)
    print "Best parameter: ", best_params
    print "Train scores for each candidate: ", train_scores
    print "Test scores for each candidate: ", test_scores
    print "RMSE score on heldout test data: ", scores
    print "Mean value of predictions: ", mean_of_predictions
    return scores, mean_of_predictions


if __name__ == '__main__':
    pass
