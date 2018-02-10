from data.transformed_data import *
from data.raw_data import data_dir
from sklearn.neighbors import RadiusNeighborsRegressor

regressor = RadiusNeighborsRegressor()
regressor.fit(train_x, train_y)

print('RadiusNeighborsRegressor rmse:{}'.format(RMSLE(validation_y, regressor.predict(validation_x))))

predict = regressor.predict(test[col])
test['visitors'] = np.expm1(predict)
test['visitors'] = test['visitors'].clip(lower=0.)
test[['id', 'visitors']].to_csv(data_dir + 'submission_radius_neighbors_regressor.csv', index=False)