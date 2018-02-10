from data.transformed_data3 import *
from data.features3 import *
from data.raw_data import data_dir
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

regressor_name = 'knn'

for test_week_of_year in test_week_of_year:
    boost_params = {'eval_metric': 'rmse'}
    regressor = KNeighborsRegressor(n_jobs=8, n_neighbors=4)
    train_x, train_y, test_x, train_data, test_data = train_and_test_other(test_week_of_year, test_cols, categorical_columns, one_hot_columns, is_numerical=False)
    predict_and_save_and_eval(regressor, train_x, train_y, test_x, train_data, test_data, regressor_name, test_week_of_year)