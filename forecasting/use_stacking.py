import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from sklearn.cross_validation import KFold
import xgboost as xgb

data_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Recruit Restaurant Visitor Forecasting\\'
data = {
    'air_visit_data': pd.read_csv(data_dir + 'air_visit_data.csv'),
    'air_store_info': pd.read_csv(data_dir + 'air_store_info.csv'),
    'hpg_store_info': pd.read_csv(data_dir + 'hpg_store_info.csv'),
    'air_reserve': pd.read_csv(data_dir + 'air_reserve.csv'),
    'hpg_reserve': pd.read_csv(data_dir + 'hpg_reserve.csv'),
    'store_id_relation': pd.read_csv(data_dir + 'store_id_relation.csv'),
    'sample_submission': pd.read_csv(data_dir + 'sample_submission.csv'),
    'date_info': pd.read_csv(data_dir + 'date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

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
data['sample_submission']['air_store_id'] = data['sample_submission']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['sample_submission']['visit_date'] = pd.to_datetime(data['sample_submission']['visit_date'])
data['sample_submission']['dow'] = data['sample_submission']['visit_date'].dt.dayofweek
data['sample_submission']['year'] = data['sample_submission']['visit_date'].dt.year
data['sample_submission']['month'] = data['sample_submission']['visit_date'].dt.month
data['sample_submission']['visit_date'] = data['sample_submission']['visit_date'].dt.date
# sample_submission: air_store_id, visit_date, dow, year, month, visitors

# 每一个饭店都有一个特征：在一周的某一天的状况
unique_stores = data['sample_submission']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i] * len(unique_stores)}) for i in range(7)],
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

col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
train = train.fillna(-1)
test = test.fillna(-1)


def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5

# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0):
        self.clf = clf

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Create 5 objects that represent our 4 models
gradient_boosting_regressor = SklearnHelper(clf=ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3), seed=SEED)
kneighbors_regressor = SklearnHelper(clf=neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=5), seed=SEED)
svr = SklearnHelper(clf=svm.SVR(C=10), seed=SEED)
rf = SklearnHelper(clf=ensemble.RandomForestRegressor(n_estimators=99, n_jobs=1), seed=SEED)
ada = SklearnHelper(clf=ensemble.AdaBoostRegressor(n_estimators=99), seed=SEED)
lr = SklearnHelper(clf=linear_model.LinearRegression(n_jobs=-1), seed=SEED)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['visitors'].values
x_train = train[col].values # Creates an array of the train data
x_test = test[col].values # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
gradient_boosting_regressor_oof_train, gradient_boosting_regressor_oof_test = get_oof(gradient_boosting_regressor, x_train, y_train, x_test)
kneighbors_regressor_oof_train, kneighbors_regressor_oof_test = get_oof(kneighbors_regressor,x_train, y_train, x_test)
svr_oof_train, svr_oof_test = get_oof(svr,x_train, y_train, x_test)
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
lr_oof_train, lr_oof_test = get_oof(lr, x_train, y_train, x_test)

x_train = np.concatenate((gradient_boosting_regressor_oof_train, kneighbors_regressor_oof_train, svr_oof_train, rf_oof_train, ada_oof_train, lr_oof_train), axis=1)
x_test = np.concatenate((gradient_boosting_regressor_oof_test, kneighbors_regressor_oof_test, svr_oof_test, rf_oof_test, ada_oof_test, lr_oof_test), axis=1)

gbm = xgb.XGBRegressor(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9, colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

# Generate Submission File
StackingSubmission = pd.DataFrame({ 'id': test['id'],
                            'visitors': predictions })
StackingSubmission.to_csv(data_dir + "StackingSubmission.csv", index=False)

