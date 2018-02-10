from data.raw_data import data
from data.raw_data import data_dir
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import *
from sklearn.preprocessing import LabelEncoder
import os
from util.common_util import *

train_file_path = data_dir + 'train.csv'
test_file_path = data_dir + 'test.csv'

def add_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df[dummies.columns] = dummies

def add_dummies_with_prefix(df, column_name, prefix):
    dummies = pd.get_dummies(df[column_name], prefix=prefix)
    df[dummies.columns] = dummies

def transform(train_or_test, reserve_visitors_groupby_visit_date, reserve_visitors_groupby_reserve_date, reserve_datetime_diff_groupby_visit_date, reserve_datetime_diff_groupby_reserve_date):
    train_or_test = pd.merge(train_or_test, data['air_store_info'], how='left', on=['air_store_id'])

    train_or_test = pd.merge(train_or_test, data['date_info'], how='left', on=['visit_date'])

    train_or_test.sort_values(by='visit_date', axis=0, ascending=True)

    train_or_test = pd.merge(train_or_test, reserve_visitors_groupby_visit_date, how='left',
                             on=['air_store_id', 'visit_date']) \
        .rename(columns={'reserve_visitors': 'reserve_visitors_in'})
    train_or_test = pd.merge(train_or_test, reserve_visitors_groupby_reserve_date, how='left',
                             left_on=['air_store_id', 'visit_date'],
                             right_on=['air_store_id', 'reserve_date']) \
        .rename(columns={'reserve_visitors': 'reserve_visitors_out'})
    train_or_test = train_or_test.drop('reserve_date', axis=1)
    train_or_test = pd.merge(train_or_test, reserve_datetime_diff_groupby_visit_date, how='left',
                             on=['air_store_id', 'visit_date']) \
        .rename(columns={'reserve_datetime_diff': 'reserve_datetime_diff_in'})
    train_or_test = pd.merge(train_or_test, reserve_datetime_diff_groupby_reserve_date, how='left',
                             left_on=['air_store_id', 'visit_date'],
                             right_on=['air_store_id', 'reserve_date']) \
        .rename(columns={'reserve_datetime_diff': 'reserve_datetime_diff_out'})
    train_or_test = train_or_test.drop('reserve_date', axis=1)

    train_or_test['visit_datetime'] = pd.to_datetime(train['visit_date'])
    train_or_test['year'] = train_or_test['visit_datetime'].dt.year
    train_or_test['month'] = train_or_test['visit_datetime'].dt.month
    train_or_test = train_or_test.drop('visit_datetime', 1)

    train_or_test = train_or_test.fillna(0)

    train_or_test['id'] = train_or_test.apply(lambda r: r['air_store_id'] + '_' + r['visit_date'], axis=1)

    return train_or_test

data['hpg_reserve'] = pd.merge(data['hpg_reserve'], data['store_id_relation'], how='inner', on=['hpg_store_id'])
data['hpg_reserve'] = data['hpg_reserve'].drop('hpg_store_id', axis=1)
for df in ['air_reserve', 'hpg_reserve']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                       axis=1)
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date.map(str)
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date.map(str)
    data[df] = data[df].rename(columns={'visit_datetime': 'visit_date', 'reserve_datetime': 'reserve_date'})

temp = pd.concat([data['air_reserve'], data['hpg_reserve']], axis=0)
'''对测试集来说，下面这四个值都不准'''
reserve_visitors_groupby_visit_date = temp.groupby(['air_store_id', 'visit_date'], as_index=False)[
    ['reserve_visitors']].sum().sort_values(by='visit_date', axis=0, ascending=True)
reserve_visitors_groupby_reserve_date = temp.groupby(['air_store_id', 'reserve_date'], as_index=False)[
    ['reserve_visitors']].sum().sort_values(by='reserve_date', axis=0, ascending=True)
reserve_datetime_diff_groupby_visit_date = temp.groupby(['air_store_id', 'visit_date'], as_index=False)[
    ['reserve_datetime_diff']].sum().sort_values(by='visit_date', axis=0, ascending=True)
reserve_datetime_diff_groupby_reserve_date = temp.groupby(['air_store_id', 'reserve_date'], as_index=False)[
    ['reserve_datetime_diff']].sum().sort_values(by='reserve_date', axis=0, ascending=True)

lbl = LabelEncoder()
for i in range(2):
    data['air_store_info']['air_genre_name_' + str(i)] = lbl.fit_transform(data['air_store_info']['air_genre_name'].map(
        lambda x: str(str(x).split()[i]) if len(str(x).split()) > i else ''))
    add_dummies(data['air_store_info'], 'air_genre_name_' + str(i))
for i in range(5):
    data['air_store_info']['air_area_name_' + str(i)] = lbl.fit_transform(data['air_store_info']['air_area_name'].map(
        lambda x: str(str(x).split()[i]) if len(str(x).split()) > i else ''))
    add_dummies(data['air_store_info'], 'air_area_name_' + str(i))
data['air_store_info']['air_genre_name'] = lbl.fit_transform(data['air_store_info']['air_genre_name'])
add_dummies_with_prefix(data['air_store_info'], 'air_genre_name', 'air_genre_name_original')
data['air_store_info']['air_area_name'] = lbl.fit_transform(data['air_store_info']['air_area_name'])
add_dummies_with_prefix(data['air_store_info'], 'air_area_name', 'air_area_name_original')

data['air_store_info']['lon_plus_lat'] = data['air_store_info']['longitude'] + data['air_store_info']['latitude']
data['air_store_info']['lon_plus_lat'] = data['air_store_info']['longitude'] + data['air_store_info']['latitude']

data['date_info']['day_of_week'] = lbl.fit_transform(data['date_info']['day_of_week'])
add_dummies(data['date_info'], 'day_of_week')

data['sample_submission']['visit_date'] = data['sample_submission']['id'].map(lambda x: str(x).split('_')[2])
data['sample_submission']['air_store_id'] = data['sample_submission']['id'].map(
    lambda x: '_'.join(x.split('_')[:2]))

train = data['air_visit_data']
train['visit_date'] = train['visit_date'].map(str)
test = data['sample_submission'].drop('id', axis=1)

train = transform(train, reserve_visitors_groupby_visit_date, reserve_visitors_groupby_reserve_date, reserve_datetime_diff_groupby_visit_date, reserve_datetime_diff_groupby_reserve_date)
test = transform(test, reserve_visitors_groupby_visit_date, reserve_visitors_groupby_reserve_date, reserve_datetime_diff_groupby_visit_date, reserve_datetime_diff_groupby_reserve_date)

temp = train.groupby(['air_store_id', 'day_of_week']).agg(
    {'visitors': [np.min, np.mean, np.median, np.max, np.size]}).reset_index()
temp.columns = ['air_store_id', 'day_of_week', 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors',
                'count_observations']
temp['max_minus_min'] = temp['max_visitors'] - temp['min_visitors']

train = pd.merge(train, temp, how='left', on=['air_store_id', 'day_of_week'])
test = pd.merge(test, temp, how='left', on=['air_store_id', 'day_of_week'])

for i in range(1, 40):
    history = str(i) + '_day_before'
    temp = train[['air_store_id', 'visit_date', 'visitors']].rename(columns={'visitors': history + '_visitors'})
    temp['visit_date'] = pd.to_datetime(temp['visit_date'])
    temp['visit_date'] = temp['visit_date'].map(lambda x: get_days_after_today(x, i)).dt.date.map(str)
    train = pd.merge(train, temp, how='left', on=['air_store_id', 'visit_date'])
    test = pd.merge(test, temp, how='left', on=['air_store_id', 'visit_date'])

train['weekofyear'] = pd.to_datetime(train['visit_date']).dt.weekofyear
test['weekofyear'] = pd.to_datetime(test['visit_date']).dt.weekofyear
for i in range(1, 8):
    history = str(i) + '_week_before'
    temp = train[['air_store_id', 'visit_date', 'visitors']].rename(columns={'visitors': history + '_visitors'})
    temp['visit_date'] = pd.to_datetime(temp['visit_date'])
    temp[history] = temp['visit_date'].map(lambda x: get_days_after_today(x, i * 7))
    temp['year'] = temp[history].dt.year
    temp['weekofyear'] = temp[history].dt.weekofyear
    temp = temp.groupby(['air_store_id', 'year', 'weekofyear']).agg(
        {history + '_visitors': [np.min, np.mean, np.median, np.max, np.size]}).reset_index()
    statistics = [ 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors',
                   'count_observations']
    statistics = [history + '_' + s for s in statistics]
    temp.columns = ['air_store_id', 'year', 'weekofyear'] + statistics
    temp[history + '_' + 'max_minus_min'] = temp[history + '_' + 'max_visitors'] - temp[history + '_' + 'min_visitors']
    train = pd.merge(train, temp, how='left', on=['air_store_id', 'year', 'weekofyear'])
    test = pd.merge(test, temp, how='left', on=['air_store_id', 'year', 'weekofyear'])

for i in range(1, 3):
    history = str(i) + '_month_before'
    temp = train[['air_store_id', 'visit_date', 'visitors']].rename(columns={'visitors': history + '_visitors'})
    temp['visit_date'] = pd.to_datetime(temp['visit_date'])
    temp[history] = temp['visit_date'].map(lambda x: get_days_after_today(x, i * 30))
    temp['year'] = temp[history].dt.year
    temp['month'] = temp[history].dt.month
    temp = temp.groupby(['air_store_id', 'year', 'month']).agg(
        {history + '_visitors': [np.min, np.mean, np.median, np.max, np.size]}).reset_index()
    statistics = [ 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors',
                   'count_observations']
    statistics = [history + '_' + s for s in statistics]
    temp.columns = ['air_store_id', 'year', 'month'] + statistics
    temp[history + '_' + 'max_minus_min'] = temp[history + '_' + 'max_visitors'] - temp[history + '_' + 'min_visitors']
    train = pd.merge(train, temp, how='left', on=['air_store_id', 'year', 'month'])
    test = pd.merge(test, temp, how='left', on=['air_store_id', 'year', 'month'])

for i in range(1, 3):
    history = str(i) + '_month_before_dayofweek'
    temp = train[['air_store_id', 'visit_date', 'day_of_week', 'visitors']].rename(columns={'visitors': history + '_visitors'})
    temp['visit_date'] = pd.to_datetime(temp['visit_date'])
    temp[history] = temp['visit_date'].map(lambda x: get_days_after_today(x, i * 30))
    temp['year'] = temp[history].dt.year
    temp['month'] = temp[history].dt.month
    temp = temp.groupby(['air_store_id', 'year', 'month', 'day_of_week']).agg(
        {history + '_visitors': [np.min, np.mean, np.median, np.max, np.size]}).reset_index()
    statistics = [ 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors',
                   'count_observations']
    statistics = [history + '_' + s for s in statistics]
    temp.columns = ['air_store_id', 'year', 'month', 'day_of_week'] + statistics
    temp[history + '_' + 'max_minus_min'] = temp[history + '_' + 'max_visitors'] - temp[history + '_' + 'min_visitors']
    train = pd.merge(train, temp, how='left', on=['air_store_id', 'year', 'month', 'day_of_week'])
    test = pd.merge(test, temp, how='left', on=['air_store_id', 'year', 'month', 'day_of_week'])

goldenweek = ['2016-04-29', '2016-04-30','2016-05-01','2016-05-02','2016-05-03','2016-05-04','2016-05-05',
              '2017-04-29','2017-04-30','2017-05-01','2017-05-02','2017-05-03','2017-05-04','2017-05-05',]
train['goldenweek'] = train.apply(lambda x : 1 if x.visit_date in goldenweek else 0, axis=1)
test['goldenweek'] = test.apply(lambda x : 1 if x.visit_date in goldenweek else 0, axis=1)

train = train.fillna(0)
test = test.fillna(0)

lbl.fit(train['air_store_id'])
train['air_store_id'] = lbl.transform(train['air_store_id'])
add_dummies(train, 'air_store_id')
test['air_store_id'] = lbl.transform(test['air_store_id'])
add_dummies(test, 'air_store_id')

add_dummies(train, 'year')
add_dummies(test, 'year')

add_dummies(train, 'month')
add_dummies(test, 'month')

add_dummies(train, 'weekofyear')
add_dummies(test, 'weekofyear')

train.sort_index(axis=1, inplace=True)
test.sort_index(axis=1, inplace=True)

numerical_columns = [
    '10_day_before_visitors',
    '11_day_before_visitors',
    '12_day_before_visitors',
    '13_day_before_visitors',
    '14_day_before_visitors',
    '15_day_before_visitors',
    '16_day_before_visitors',
    '17_day_before_visitors',
    '18_day_before_visitors',
    '19_day_before_visitors',
    '1_day_before_visitors',
    '1_month_before_count_observations',
    '1_month_before_dayofweek_count_observations',
    '1_month_before_dayofweek_max_minus_min',
    '1_month_before_dayofweek_max_visitors',
    '1_month_before_dayofweek_mean_visitors',
    '1_month_before_dayofweek_median_visitors',
    '1_month_before_dayofweek_min_visitors',
    '1_month_before_max_minus_min',
    '1_month_before_max_visitors',
    '1_month_before_mean_visitors',
    '1_month_before_median_visitors',
    '1_month_before_min_visitors',
    '1_week_before_count_observations',
    '1_week_before_max_minus_min',
    '1_week_before_max_visitors',
    '1_week_before_mean_visitors',
    '1_week_before_median_visitors',
    '1_week_before_min_visitors',
    '20_day_before_visitors',
    '21_day_before_visitors',
    '22_day_before_visitors',
    '23_day_before_visitors',
    '24_day_before_visitors',
    '25_day_before_visitors',
    '26_day_before_visitors',
    '27_day_before_visitors',
    '28_day_before_visitors',
    '29_day_before_visitors',
    '2_day_before_visitors',
    '2_month_before_count_observations',
    '2_month_before_dayofweek_count_observations',
    '2_month_before_dayofweek_max_minus_min',
    '2_month_before_dayofweek_max_visitors',
    '2_month_before_dayofweek_mean_visitors',
    '2_month_before_dayofweek_median_visitors',
    '2_month_before_dayofweek_min_visitors',
    '2_month_before_max_minus_min',
    '2_month_before_max_visitors',
    '2_month_before_mean_visitors',
    '2_month_before_median_visitors',
    '2_month_before_min_visitors',
    '2_week_before_count_observations',
    '2_week_before_max_minus_min',
    '2_week_before_max_visitors',
    '2_week_before_mean_visitors',
    '2_week_before_median_visitors',
    '2_week_before_min_visitors',
    '30_day_before_visitors',
    '31_day_before_visitors',
    '32_day_before_visitors',
    '33_day_before_visitors',
    '34_day_before_visitors',
    '35_day_before_visitors',
    '36_day_before_visitors',
    '37_day_before_visitors',
    '38_day_before_visitors',
    '39_day_before_visitors',
    '3_day_before_visitors',
    '3_week_before_count_observations',
    '3_week_before_max_minus_min',
    '3_week_before_max_visitors',
    '3_week_before_mean_visitors',
    '3_week_before_median_visitors',
    '3_week_before_min_visitors',
    '4_day_before_visitors',
    '4_week_before_count_observations',
    '4_week_before_max_minus_min',
    '4_week_before_max_visitors',
    '4_week_before_mean_visitors',
    '4_week_before_median_visitors',
    '4_week_before_min_visitors',
    '5_day_before_visitors',
    '5_week_before_count_observations',
    '5_week_before_max_minus_min',
    '5_week_before_max_visitors',
    '5_week_before_mean_visitors',
    '5_week_before_median_visitors',
    '5_week_before_min_visitors',
    '6_day_before_visitors',
    '6_week_before_count_observations',
    '6_week_before_max_minus_min',
    '6_week_before_max_visitors',
    '6_week_before_mean_visitors',
    '6_week_before_median_visitors',
    '6_week_before_min_visitors',
    '7_day_before_visitors',
    '7_week_before_count_observations',
    '7_week_before_max_minus_min',
    '7_week_before_max_visitors',
    '7_week_before_mean_visitors',
    '7_week_before_median_visitors',
    '7_week_before_min_visitors',
    '8_day_before_visitors',
    '9_day_before_visitors',
    'latitude',
    'lon_plus_lat',
    'longitude',
    'max_minus_min',
    'max_visitors',
    'mean_visitors',
    'median_visitors',
    'min_visitors',
]
# 规范化数值型数据
for numerical_column in numerical_columns:
    column = train[numerical_column]
    mean = column.mean()
    std = column.std()
    train[numerical_column] = (train[numerical_column] - mean) / std
    test[numerical_column] = (test[numerical_column] - mean) / std

for i in range(16, 23):
    last_dot_index = test_file_path.rfind('.')
    test_part_path = test_file_path[:last_dot_index]
    test_part_path += '_'
    test_part_path += str(i)
    test_part_path += test_file_path[last_dot_index:]
    test_part_data = test[test['weekofyear'] == i]
    test_part_data.to_csv(test_part_path, index=False, encoding='utf-8')

train.to_csv(train_file_path, index=False, encoding='utf-8')
test.to_csv(test_file_path, index=False, encoding='utf-8')
