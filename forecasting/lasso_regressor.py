from data.transformed_data3 import *
from data.features3 import *
from data.raw_data import data_dir
from xgboost import XGBRegressor
from sklearn import *
from sklearn.linear_model import Lasso

regressor_name = 'lasso'

for test_week_of_year in test_week_of_year:
    regressor = Lasso(alpha=0.01, random_state=1234)
    train_x, train_y, test_x, train_data, test_data = train_and_test(test_week_of_year, test_cols, categorical_columns, one_hot_columns, is_numerical=True)
    predict_and_save_and_eval(regressor, train_x, train_y, test_x, train_data, test_data, regressor_name, test_week_of_year)