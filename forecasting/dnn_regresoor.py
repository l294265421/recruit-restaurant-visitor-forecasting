from data.transformed_data3 import *
from data.features3 import *
from data.raw_data import data_dir
from forecasting.dnn_wrapper import DnnRegressor

regressor_name = 'dnn'

for test_week_of_year in [22]:
    col = test_cols[test_week_of_year]
    numerical_columns = [column for column in col if column not in categorical_columns]
    numerical_columns = [column for column in numerical_columns if column not in one_hot_columns]
    regressor = DnnRegressor(numerical_columns, categorical_columns, mode_type='deep', train_epochs=15)
    train_x, train_y, test_x, train_data, test_data = train_and_test(test_week_of_year, test_cols, categorical_columns, one_hot_columns, is_numerical=False)
    predict_and_save_and_eval(regressor, train_x, train_y, test_x, train_data, test_data, regressor_name, test_week_of_year, dnn=True, categorical_columns=categorical_columns)