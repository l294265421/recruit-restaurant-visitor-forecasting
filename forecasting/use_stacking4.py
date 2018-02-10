from data.raw_data import data_dir
from data.transformed_data3 import RMSLE
from data.transformed_data3 import validation_predict_dir_prefix
from data.transformed_data3 import test_predict_dir_prefix
from data.features3 import *
from sklearn import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import os

for week in test_week_of_year:
    validation_predict_dir = validation_predict_dir_prefix + str(week) + '\\'
    test_predict_dir = test_predict_dir_prefix + str(week) + '\\'
    validation_predicts = []
    test_predicts = []
    validation_paths = os.listdir(validation_predict_dir)
    for p in validation_paths:
        validation_predicts.append(pd.read_csv(validation_predict_dir + p)['visitors_predict'].tolist())
        validation_predicts_y = np.array(pd.read_csv(validation_predict_dir + p)['visitors'].tolist())

    test_paths = os.listdir(test_predict_dir)
    for p in test_paths:
        test = pd.read_csv(test_predict_dir + p)
        test_predicts.append(test['visitors'].tolist())

    validation_predicts = [np.array(predicts).reshape(-1, 1) for predicts in validation_predicts]
    test_predicts = [np.array(predicts).reshape(-1, 1) for predicts in test_predicts]

    train_merge = np.concatenate(validation_predicts, axis=1)
    test_merge = np.concatenate(test_predicts, axis=1)

    predict = np.mean(test_merge, axis=1)

    test['visitors'] = predict
    test['visitors'] = test['visitors'].clip(lower=0.)
    test[['id', 'visitors']].to_csv(data_dir + 'submission_use_stacking4_' + str(week) + '.csv', index=False)