from data.raw_data import data
from data.raw_data import data_dir
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import *
import os
from sklearn.preprocessing import StandardScaler
from forecasting.stacking_regressor import stacking_predict

validation_predict_dir_prefix = data_dir + 'validation_predict_dir\\'

test_predict_dir_prefix = data_dir + 'test_predict_dir\\'

SEED = 1234
NFOLDS = 3

def train_and_test(test_week_of_year, test_cols, categorical_columns, one_hot_columns, is_numerical=True):
    train_file_path = data_dir + 'train.csv'
    test_file_path = data_dir + 'test_' + str(test_week_of_year) + '.csv'

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    # 过滤掉历史信息不足的样本
    train_keep_flag = train_data.apply(lambda x: not (x['year'] == 2016 and (x['month'] == 1 or x['month'] == 2)), axis=1)
    train_data = train_data[train_keep_flag].reset_index(drop=True)

    col = test_cols[test_week_of_year]
    if is_numerical:
        col = [column for column in col if column not in categorical_columns]
    else:
        col = [column for column in col if column not in one_hot_columns]

    train_x = train_data[col]
    train_y = np.log1p(train_data['visitors'].values)
    test_x = test_data[col]

    train_data = train_data[['id', 'visitors']]
    test_data = test_data[['id']]

    return train_x, train_y, test_x, train_data, test_data

def train_and_test_other(test_week_of_year, test_cols, categorical_columns, one_hot_columns, is_numerical=True):
    train_file_path = data_dir + 'train_other.csv'
    test_file_path = data_dir + 'test_' + str(test_week_of_year) + '_other.csv'

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    # 与train_and_test方法得到的训练样本保持一致
    train_keep_flag = train_data.apply(lambda x: not (x['year_for_filter'] == 2016 and (x['month_for_filter'] == 1 or x['month_for_filter'] == 2)), axis=1)
    train_data = train_data[train_keep_flag].reset_index(drop=True)

    col = [c for c in train_data if c not in ['id', 'air_store_id', 'visit_date', 'visitors', 'weekofyear', 'year_for_filter', 'month_for_filter']]

    train_x = train_data[col]
    train_y = np.log1p(train_data['visitors'].values)
    test_x = test_data[col]

    train_data = train_data[['id', 'visitors']]
    test_data = test_data[['id']]

    return train_x, train_y, test_x, train_data, test_data

def predict_and_save_and_eval(regressor, train_x, train_y, test_x, train_data, test_data, regressor_name, test_week_of_year, dnn=False, categorical_columns = []):
    oof_train, oof_test = stacking_predict(regressor, train_x, train_y, test_x, NFOLDS, SEED, dnn=dnn, categorical_columns = categorical_columns)
    train_data = train_data.copy()
    train_data['visitors_predict'] = np.expm1(oof_train)
    train_data['visitors_predict'] = train_data['visitors_predict'].clip(lower=0.)
    if not os.path.exists(validation_predict_dir_prefix + str(test_week_of_year)):
        os.makedirs(validation_predict_dir_prefix + str(test_week_of_year))
    rmsle = RMSLE(train_y, oof_train)
    print(regressor_name + ' rmse:{}'.format(rmsle))
    train_data[['id', 'visitors', 'visitors_predict']].to_csv(validation_predict_dir_prefix + str(test_week_of_year) + '\\' + regressor_name + '_' + str(rmsle) + '.csv', index=False)

    test_data = test_data.copy()
    test_data['visitors'] = np.expm1(oof_test)
    test_data['visitors'] = test_data['visitors'].clip(lower=0.)
    if not os.path.exists(test_predict_dir_prefix + str(test_week_of_year)):
        os.makedirs(test_predict_dir_prefix + str(test_week_of_year))
    test_data[['id', 'visitors']].to_csv(
            test_predict_dir_prefix + str(test_week_of_year) + '\\' + regressor_name + '_' + str(rmsle) + '.csv',
            index=False)

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5