from data.transformed_data2 import *
from data.raw_data import data_dir
from sklearn import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor

class SklearnHelper(object):
    def __init__(self, regressor, seed=0):
        self.regressor = regressor

    def train(self, x_train, y_train):
        self.regressor.fit(x_train, y_train)

    def predict(self, x):
        return self.regressor.predict(x)

    def fit(self, x, y):
        return self.regressor.fit(x, y)

    def feature_importances(self, x, y):
        print(self.regressor.fit(x, y).feature_importances_)

def get_predicts(regressor, train_x, train_y, validation_x, test):
    regressor.fit(train_x, train_y)
    return (regressor.predict(validation_x).reshape(-1, 1), regressor.predict(test).reshape(-1, 1))

SEED = 1234
gradient_boosting_regressor = SklearnHelper(ensemble.GradientBoostingRegressor(learning_rate=0.1, random_state=1234,
                                      n_estimators=999,
                                      min_samples_split=5,
                                      min_samples_leaf=3,
                                      max_features='sqrt',
                                      subsample=0.8), seed=SEED)
# gradient_boosting_regressor = SklearnHelper(XGBRegressor(n_jobs=-1,  **{'eval_metric': 'rmse'}), seed=SEED)
kneighbors_regressor = SklearnHelper(neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=7, weights='distance'), seed=SEED)
svr = SklearnHelper(svm.SVR(C=10), seed=SEED)
rf = SklearnHelper(ensemble.RandomForestRegressor(n_estimators=99, n_jobs=1), seed=SEED)
ada = SklearnHelper(ensemble.AdaBoostRegressor(n_estimators=99), seed=SEED)

gradient_boosting_regressor_oof_train, gradient_boosting_regressor_oof_test = get_predicts(gradient_boosting_regressor, train_x, train_y, validation_x, test_data)
print('gradient_boosting_regressor complete')
kneighbors_regressor_oof_train, kneighbors_regressor_oof_test = get_predicts(kneighbors_regressor, train_x, train_y, validation_x, test_data)
print('kneighbors_regressor complete')
# svr_oof_train, svr_oof_test = get_predicts(svr, train_x, train_y, validation_x, test_data)
# print('svr complete')
rf_oof_train, rf_oof_test = get_predicts(rf, train_x, train_y, validation_x, test_data)
print('rf complete')
ada_oof_train, ada_oof_test = get_predicts(ada, train_x, train_y, validation_x, test_data)
print('ada complete')

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5

print('RMSE gradient_boosting_regressor: ', RMSLE(validation_y, gradient_boosting_regressor_oof_train))
print('RMSE kneighbors_regressor_oof_train: ', RMSLE(validation_y, kneighbors_regressor_oof_train))
print('RMSE rf: ', RMSLE(validation_y, rf_oof_train))
print('RMSE ada: ', RMSLE(validation_y, ada_oof_train))

train_merge = np.concatenate((gradient_boosting_regressor_oof_train, kneighbors_regressor_oof_train, rf_oof_train, ada_oof_train), axis=1)
test_merge = np.concatenate((gradient_boosting_regressor_oof_test, kneighbors_regressor_oof_test, rf_oof_test, ada_oof_test), axis=1)

boost_params = {'eval_metric': 'rmse'}
lr = Ridge(random_state=1234)

lr.fit(train_merge, validation_y)
predict = lr.predict(test_merge)
temp = lr.predict(train_merge)
print('RMSE stacking: ', RMSLE(validation_y.reshape(-1, 1), lr.predict(train_merge)))
test['visitors'] = np.expm1(predict)
test['visitors'] = test['visitors'].clip(lower=0.)
test[['id', 'visitors']].to_csv(data_dir + 'submission_use_stacking2_' + str(test_week_of_year) + '.csv', index=False)


