from data.transformed_data3 import *
from data.features3 import *
from data.raw_data import data_dir
from sklearn.ensemble import ExtraTreesRegressor

regressor_name = 'extratree'

for test_week_of_year in test_week_of_year:
    regressor = ExtraTreesRegressor(n_estimators=999, max_depth=8, n_jobs=-1)
    train_x, train_y, test_x, train_data, test_data = train_and_test(test_week_of_year, test_cols, categorical_columns, one_hot_columns, is_numerical=False)
    predict_and_save_and_eval(regressor, train_x, train_y, test_x, train_data, test_data, regressor_name, test_week_of_year)