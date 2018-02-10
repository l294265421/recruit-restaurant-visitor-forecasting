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
from sklearn.ensemble import RandomForestRegressor
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
        validation_predicts_y = np.log1p(np.array(pd.read_csv(validation_predict_dir + p)['visitors'].tolist()))

    test_paths = os.listdir(test_predict_dir)
    for p in test_paths:
        test = pd.read_csv(test_predict_dir + p)
        test_predicts.append(test['visitors'].tolist())

    validation_predicts = [np.array(predicts).reshape(-1, 1) for predicts in validation_predicts]
    test_predicts = [np.array(predicts).reshape(-1, 1) for predicts in test_predicts]

    train_merge = np.concatenate(validation_predicts, axis=1)
    test_merge = np.concatenate(test_predicts, axis=1)

    # lr = Ridge(random_state=1234, alpha=0.01, normalize=True)
    # lr = RandomForestRegressor(n_estimators=200, n_jobs=-1, max_depth=7, random_state=1234)
    lr = XGBRegressor(subsample=0.7, colsample_bytree=0.8, n_jobs=-1, random_state=1234, reg_lambda=0.01, reg_alpha=0.01, n_estimators=200, max_depth=7)
    # lr = SVR(C=10)
    lr.fit(train_merge, validation_predicts_y)
    predict = lr.predict(test_merge)
    temp = lr.predict(train_merge)
    print('RMSE stacking: ', RMSLE(validation_predicts_y, lr.predict(train_merge)))
    test['visitors'] = np.expm1(predict)
    test['visitors'] = test['visitors'].clip(lower=0.)
    test[['id', 'visitors']].to_csv(data_dir + 'submission_use_stacking3_' + str(week) + '.csv', index=False)