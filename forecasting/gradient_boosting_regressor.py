from data.transformed_data2 import *
from data.raw_data import data_dir
from sklearn.ensemble import GradientBoostingRegressor

regressor_name = 'gradient_bootint'
regressor = GradientBoostingRegressor(learning_rate=0.1, random_state=1234,
                                      n_estimators=999,
                                      min_samples_split=5,
                                      min_samples_leaf=3,
                                      max_features='sqrt',
                                      subsample=0.8)
regressor.fit(train_x, train_y)

predict_and_save_and_eval(regressor, train_x, train_y, train, train_predict_dir_prefix, regressor_name, test_week_of_year)
predict_and_save_and_eval(regressor, validation_x, validation_y, validation, validation_predict_dir_prefix, regressor_name, test_week_of_year)

predict_and_save(regressor, test_x, test, test_predict_dir_prefix, regressor_name, test_week_of_year)