from data.transformed_data3 import *
from data.features3 import *
from data.raw_data import data_dir
from xgboost import XGBRegressor

regressor_name = 'xgboost'

for test_week_of_year in test_week_of_year:
    boost_params = {'eval_metric': 'rmse'}
    regressor = XGBRegressor(
        max_depth=8,
        learning_rate=0.01,
        n_estimators=250,
        objective='reg:linear',
        gamma=0,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        seed=27,
        **boost_params)
    train_x, train_y, test_x, train_data, test_data = train_and_test(test_week_of_year, test_cols, categorical_columns, one_hot_columns, is_numerical=False)
    predict_and_save_and_eval(regressor, train_x, train_y, test_x, train_data, test_data, regressor_name, test_week_of_year)