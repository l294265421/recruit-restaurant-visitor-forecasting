from data.transformed_data3 import *
from data.features3 import *
from data.raw_data import data_dir
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

regressor_name = 'gredient_boosting_other'

for test_week_of_year in test_week_of_year:
    boost_params = {'eval_metric': 'rmse'}
    regressor = GradientBoostingRegressor(learning_rate=0.2, random_state=3,
                    n_estimators=200, subsample=0.8, max_depth =10)
    train_x, train_y, test_x, train_data, test_data = train_and_test_other(test_week_of_year, test_cols, categorical_columns, one_hot_columns, is_numerical=False)
    predict_and_save_and_eval(regressor, train_x, train_y, test_x, train_data, test_data, regressor_name, test_week_of_year)