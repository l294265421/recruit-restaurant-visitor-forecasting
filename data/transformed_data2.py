from data.raw_data import data
from data.raw_data import data_dir
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import *
import os
from sklearn.preprocessing import StandardScaler

test_22_col = [
 # 'visitors',
 # 'visit_date',
 'air_store_id',
 'air_genre_name',
 'air_area_name',
 'latitude',
 'longitude',
 'air_genre_name_0',
 'air_genre_name_1',
 'air_area_name_0',
 'air_area_name_1',
 'air_area_name_2',
 'air_area_name_3',
 'air_area_name_4',
 'lon_plus_lat',
 'day_of_week',
 'holiday_flg',
 # 'reserve_visitors_in',
 # 'reserve_visitors_out',
 # 'reserve_datetime_diff_in',
 # 'reserve_datetime_diff_out',
 'year',
 'month',
 # 'id',
 'min_visitors',
 'mean_visitors',
 'median_visitors',
 'max_visitors',
 'count_observations',
 # '1_day_before_visitors',
 # '2_day_before_visitors',
 # '3_day_before_visitors',
 # '4_day_before_visitors',
 # '5_day_before_visitors',
 # '6_day_before_visitors',
 # '7_day_before_visitors',
 # '8_day_before_visitors',
 # '9_day_before_visitors',
 # '10_day_before_visitors',
 # '11_day_before_visitors',
 # '12_day_before_visitors',
 # '13_day_before_visitors',
 # '14_day_before_visitors',
 # '15_day_before_visitors',
 # '16_day_before_visitors',
 # '17_day_before_visitors',
 # '18_day_before_visitors',
 # '19_day_before_visitors',
 # '20_day_before_visitors',
 # '21_day_before_visitors',
 # '22_day_before_visitors',
 # '23_day_before_visitors',
 # '24_day_before_visitors',
 # '25_day_before_visitors',
 # '26_day_before_visitors',
 # '27_day_before_visitors',
 # '28_day_before_visitors',
 # '29_day_before_visitors',
 # '30_day_before_visitors',
 # '31_day_before_visitors',
 # '32_day_before_visitors',
 # '33_day_before_visitors',
 # '34_day_before_visitors',
 # '35_day_before_visitors',
 # '36_day_before_visitors',
 # '37_day_before_visitors',
 # '38_day_before_visitors',
 '39_day_before_visitors',
 'weekofyear',
 # '1_week_before_min_visitors',
 # '1_week_before_mean_visitors',
 # '1_week_before_median_visitors',
 # '1_week_before_max_visitors',
 # '1_week_before_count_observations',
 # '2_week_before_min_visitors',
 # '2_week_before_mean_visitors',
 # '2_week_before_median_visitors',
 # '2_week_before_max_visitors',
 # '2_week_before_count_observations',
 # '3_week_before_min_visitors',
 # '3_week_before_mean_visitors',
 # '3_week_before_median_visitors',
 # '3_week_before_max_visitors',
 # '3_week_before_count_observations',
 # '4_week_before_min_visitors',
 # '4_week_before_mean_visitors',
 # '4_week_before_median_visitors',
 # '4_week_before_max_visitors',
 # '4_week_before_count_observations',
 # '5_week_before_min_visitors',
 # '5_week_before_mean_visitors',
 # '5_week_before_median_visitors',
 # '5_week_before_max_visitors',
 # '5_week_before_count_observations',
 '6_week_before_min_visitors',
 '6_week_before_mean_visitors',
 '6_week_before_median_visitors',
 '6_week_before_max_visitors',
 '6_week_before_count_observations',
 '7_week_before_min_visitors',
 '7_week_before_mean_visitors',
 '7_week_before_median_visitors',
 '7_week_before_max_visitors',
 '7_week_before_count_observations',
 '1_month_before_min_visitors',
 '1_month_before_mean_visitors',
 '1_month_before_median_visitors',
 '1_month_before_max_visitors',
 '1_month_before_count_observations',
 '2_month_before_min_visitors',
 '2_month_before_mean_visitors',
 '2_month_before_median_visitors',
 '2_month_before_max_visitors',
 '2_month_before_count_observations',
 '1_month_before_dayofweek_min_visitors',
 '1_month_before_dayofweek_mean_visitors',
 '1_month_before_dayofweek_median_visitors',
 '1_month_before_dayofweek_max_visitors',
 '1_month_before_dayofweek_count_observations',
 '2_month_before_dayofweek_min_visitors',
 '2_month_before_dayofweek_mean_visitors',
 '2_month_before_dayofweek_median_visitors',
 '2_month_before_dayofweek_max_visitors',
 '2_month_before_dayofweek_count_observations',
 'goldenweek']
test_21_col = test_22_col + [
 '36_day_before_visitors',
 '37_day_before_visitors',
 '5_week_before_min_visitors',
 '5_week_before_mean_visitors',
 '5_week_before_median_visitors',
 '5_week_before_max_visitors',
 '5_week_before_count_observations',
]
test_20_col = test_21_col + [
 '29_day_before_visitors',
 '30_day_before_visitors',
 '31_day_before_visitors',
 '32_day_before_visitors',
 '33_day_before_visitors',
 '34_day_before_visitors',
 '35_day_before_visitors',
 '4_week_before_min_visitors',
 '4_week_before_mean_visitors',
 '4_week_before_median_visitors',
 '4_week_before_max_visitors',
 '4_week_before_count_observations',
]
test_19_col = test_20_col + [
 '22_day_before_visitors',
 '23_day_before_visitors',
 '24_day_before_visitors',
 '25_day_before_visitors',
 '26_day_before_visitors',
 '27_day_before_visitors',
 '28_day_before_visitors',
 '3_week_before_min_visitors',
 '3_week_before_mean_visitors',
 '3_week_before_median_visitors',
 '3_week_before_max_visitors',
 '3_week_before_count_observations',
]
test_18_col = test_19_col + [
 '15_day_before_visitors',
 '16_day_before_visitors',
 '17_day_before_visitors',
 '18_day_before_visitors',
 '19_day_before_visitors',
 '20_day_before_visitors',
 '21_day_before_visitors',
 '2_week_before_min_visitors',
 '2_week_before_mean_visitors',
 '2_week_before_median_visitors',
 '2_week_before_max_visitors',
 '2_week_before_count_observations',
]
test_17_col = test_18_col + [
 '8_day_before_visitors',
 '9_day_before_visitors',
 '10_day_before_visitors',
 '11_day_before_visitors',
 '12_day_before_visitors',
 '13_day_before_visitors',
 '14_day_before_visitors',
 '1_week_before_min_visitors',
 '1_week_before_mean_visitors',
 '1_week_before_median_visitors',
 '1_week_before_max_visitors',
 '1_week_before_count_observations',
]
test_16_col = test_17_col + [
 '1_day_before_visitors',
 '2_day_before_visitors',
 '3_day_before_visitors',
 '4_day_before_visitors',
 '5_day_before_visitors',
 '6_day_before_visitors',
 '7_day_before_visitors',
]

test_week_of_year = 21

train_file_path = data_dir + 'train.csv'
test_file_path = data_dir + 'test_' + str(test_week_of_year) + '.csv'

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

train['goldenweek'] = train['goldenweek'].astype(int)
test['goldenweek'] = test['goldenweek'].astype(int)

test_cols = {16:test_16_col, 17:test_17_col, 18:test_18_col, 19:test_19_col, 20:test_20_col, 21:test_21_col, 22:test_22_col}

col = test_cols[test_week_of_year]

categorical_columns = [
 'air_store_id',
 'air_genre_name',
 'air_area_name',
 'air_genre_name_0',
 'air_genre_name_1',
 'air_area_name_0',
 'air_area_name_1',
 'air_area_name_2',
 'air_area_name_3',
 'air_area_name_4',
 'day_of_week',
 'year',
 'month',
 'weekofyear',
]
numerical_columns =[column for column in col if column not in categorical_columns]

dont_need_normalized =['holiday_flg', 'goldenweek']

need_normalized = [column for column in numerical_columns if column not in dont_need_normalized]

# 规范化数值型数据
for numerical_column in need_normalized:
     column = train[numerical_column]
     mean = column.mean()
     std = column.std()
     train[numerical_column] = (train[numerical_column] - mean) / std
     test[numerical_column] = (test[numerical_column] - mean) / std

ont_hot_col = []
for categorical_column in categorical_columns:
    train[categorical_column] = train[categorical_column].astype(int)
    train_one_hot = pd.get_dummies(train[categorical_column], prefix=categorical_column)
    train[train_one_hot.columns] = train_one_hot
    test_one_hot = pd.get_dummies(test[categorical_column], prefix=categorical_column)
    test[test_one_hot.columns] = test_one_hot
    ont_hot_col += list(test_one_hot.columns)

all_numerical_columns = numerical_columns + ont_hot_col

test_x = test[col]

train_keep_flag = train.apply(lambda x : not (x['year'] == 2016 and (x['month'] == 1 or x['month'] == 2)), axis = 1)
train = train[train_keep_flag]

validation_flag = train.apply(lambda x : x['year'] == 2017 and x['month'] == 4, axis = 1)
train_flag = validation_flag.map(lambda x : not x)

validation = train[validation_flag]
train = train[train_flag]


train_x = train[col]
train_y = np.log1p(train['visitors'].values)

validation_x = validation[col]
validation_y = np.log1p(validation['visitors'].values)

numerical_test_x = test[numerical_columns]

all_numerical_test_x = test[all_numerical_columns]

numerical_train_x = train[numerical_columns]

all_numerical_train_x = train[all_numerical_columns]

numerical_validation_x = validation[numerical_columns]

all_numerical_validation_x = validation[all_numerical_columns]

categorical_test_x = test[categorical_columns]

categorical_train_x = train[categorical_columns]

categorical_validation_x = validation[categorical_columns]

train_predict_dir_prefix = data_dir + 'train_predict_dir\\'

validation_predict_dir_prefix = data_dir + 'validation_predict_dir\\'

test_predict_dir_prefix = data_dir + 'test_predict_dir\\'

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5

def predict_and_save(regressor, x, test, dir_prefix, regressor_name, test_week_of_year, validation_rmsle):
    predict = regressor.predict(x)
    test['visitors'] = np.expm1(predict)
    test['visitors'] = test['visitors'].clip(lower=0.)
    if not os.path.exists(dir_prefix + str(test_week_of_year)):
        os.makedirs(dir_prefix + str(test_week_of_year))
    test[['id', 'visitors']].to_csv(dir_prefix + str(test_week_of_year) + '\\' + regressor_name + '_' + str(validation_rmsle) + '.csv',
                                index=False)

def predict_and_save_and_eval(regressor, x, y, data, dir_prefix, regressor_name, test_week_of_year):
    predit = regressor.predict(x)
    data = data.copy()
    data['visitors_predict'] = np.expm1(predit)
    data['visitors_predict'] = data['visitors_predict'].clip(lower=0.)
    if not os.path.exists(dir_prefix + str(test_week_of_year)):
        os.makedirs(dir_prefix + str(test_week_of_year))
    rmsle = RMSLE(y, predit)
    print(regressor_name + ' rmse:{}'.format(rmsle))
    data[['id', 'visitors', 'visitors_predict']].to_csv(dir_prefix + str(test_week_of_year) + '\\' + regressor_name + '_' + str(rmsle) + '.csv', index=False)
    return rmsle