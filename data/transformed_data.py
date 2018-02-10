from data.raw_data import data
from data.raw_data import data_dir
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import *
import os

train_file_path = data_dir + 'train.csv'
test_file_path = data_dir + 'test.csv'
if not os.path.exists(train_file_path)  or not os.path.exists(test_file_path):
    data['hpg_reserve'] = pd.merge(data['hpg_reserve'], data['store_id_relation'], how='inner', on=['hpg_store_id'])

    for df in ['air_reserve', 'hpg_reserve']:
        data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
        data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
        data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
        data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
        data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                           axis=1)
        tmp1 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
            ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(
            columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
        tmp2 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
            ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(
            columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
        data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id', 'visit_date'])
    # air_reserve hpg_reserve: air_store_id, visit_date, rs1, rv1, rs2, rv2

    data['air_visit_data']['visit_date'] = pd.to_datetime(data['air_visit_data']['visit_date'])
    data['air_visit_data']['dow'] = data['air_visit_data']['visit_date'].dt.dayofweek
    data['air_visit_data']['year'] = data['air_visit_data']['visit_date'].dt.year
    data['air_visit_data']['month'] = data['air_visit_data']['visit_date'].dt.month
    data['air_visit_data']['visit_date'] = data['air_visit_data']['visit_date'].dt.date
    # air_visit_data: air_store_id, visit_date, dow, year, month, visitors

    data['sample_submission']['visit_date'] = data['sample_submission']['id'].map(lambda x: str(x).split('_')[2])
    data['sample_submission']['air_store_id'] = data['sample_submission']['id'].map(
        lambda x: '_'.join(x.split('_')[:2]))
    data['sample_submission']['visit_date'] = pd.to_datetime(data['sample_submission']['visit_date'])
    data['sample_submission']['dow'] = data['sample_submission']['visit_date'].dt.dayofweek
    data['sample_submission']['year'] = data['sample_submission']['visit_date'].dt.year
    data['sample_submission']['month'] = data['sample_submission']['visit_date'].dt.month
    data['sample_submission']['visit_date'] = data['sample_submission']['visit_date'].dt.date
    # sample_submission: air_store_id, visit_date, dow, year, month, visitors

    # 每一个饭店都有一个特征：在一周的某一天的状况
    unique_stores = data['sample_submission']['air_store_id'].unique()
    stores = pd.concat(
        [pd.DataFrame({'air_store_id': unique_stores, 'dow': [i] * len(unique_stores)}) for i in range(7)],
        axis=0, ignore_index=True).reset_index(drop=True)
    # stores: air_store_id dow

    # OPTIMIZED BY JEROME VALLET
    tmp = data['air_visit_data'].groupby(['air_store_id', 'dow']).agg(
        {'visitors': [np.min, np.mean, np.median, np.max, np.size]}).reset_index()
    tmp.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors',
                   'count_observations']
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
    # stores: air_store_id, dow, min_visitors, mean_visitors, median_visitors, max_visitors, count_observations

    stores = pd.merge(stores, data['air_store_info'], how='left', on=['air_store_id'])
    # stores: air_store_id, dow, min_visitors, mean_visitors, median_visitors, max_visitors, count_observations, air_genre_name, air_area_name, latitude, longitude


    # NEW FEATURES FROM Georgii Vyshnia
    stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
    stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))
    lbl = preprocessing.LabelEncoder()
    for i in range(10):
        stores['air_genre_name' + str(i)] = lbl.fit_transform(
            stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
        stores['air_area_name' + str(i)] = lbl.fit_transform(
            stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' ')) > i else ''))
    stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
    stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
    # stores: air_store_id, dow, min_visitors, mean_visitors, median_visitors, max_visitors, count_observations, air_genre_name, air_area_name, latitude, longitude, air_genre_name0-air_genre_name9

    data['date_info']['visit_date'] = pd.to_datetime(data['date_info']['visit_date'])
    data['date_info']['day_of_week'] = lbl.fit_transform(data['date_info']['day_of_week'])
    data['date_info']['visit_date'] = data['date_info']['visit_date'].dt.date
    # date_info: visit_date, day_of_week, holiday_flg

    train = pd.merge(data['air_visit_data'], data['date_info'], how='left', on=['visit_date'])
    # train: air_store_id, visit_date, dow, year, month, visitors, day_of_week, holiday_flg
    test = pd.merge(data['sample_submission'], data['date_info'], how='left', on=['visit_date'])
    # test: air_store_id, visit_date, dow, year, month, visitors, day_of_week, holiday_flg

    train = pd.merge(train, stores, how='left', on=['air_store_id', 'dow'])
    test = pd.merge(test, stores, how='left', on=['air_store_id', 'dow'])

    for df in ['air_reserve', 'hpg_reserve']:
        train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date'])
        test = pd.merge(test, data[df], how='left', on=['air_store_id', 'visit_date'])

    train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

    train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
    train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
    train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

    test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
    test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
    test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

    # NEW FEATURES FROM JMBULL
    train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    train['var_max_lat'] = train['latitude'].max() - train['latitude']
    train['var_max_long'] = train['longitude'].max() - train['longitude']
    test['var_max_lat'] = test['latitude'].max() - test['latitude']
    test['var_max_long'] = test['longitude'].max() - test['longitude']

    # NEW FEATURES FROM Georgii Vyshnia
    train['lon_plus_lat'] = train['longitude'] + train['latitude']
    test['lon_plus_lat'] = test['longitude'] + test['latitude']

    lbl = preprocessing.LabelEncoder()
    train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
    test['air_store_id2'] = lbl.transform(test['air_store_id'])

    train = train.fillna(-1)
    test = test.fillna(-1)
    col = [c for c in train]
    train = train[col]
    test = test[col]
    train.to_csv(train_file_path, index=False)
    test.to_csv(test_file_path, index=False)

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)
col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
test_data = test[col]

validation_flag = train.apply(lambda x : x['year'] == 2017 and x['month'] == 4, axis = 1)
train_flag = validation_flag.map(lambda x : not x)

train_data = train[train_flag]
validation_data = train[validation_flag]

train_x = train_data[col]
train_y = np.log1p(train_data['visitors'].values)

validation_x = validation_data[col]
validation_y = np.log1p(validation_data['visitors'].values)

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5