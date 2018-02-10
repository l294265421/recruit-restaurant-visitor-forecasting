import numpy as np
from sklearn.cross_validation import KFold

def stacking_predict(regressor, train_x, train_y, test_x, nflolds, seed, dnn=False, categorical_columns = []):
    if not dnn:
        train_x = train_x.values
        test_x = test_x.values
    ntrain = train_x.shape[0]
    ntest = test_x.shape[0]
    kf = KFold(ntrain, n_folds=nflolds, random_state=seed)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((nflolds, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        if dnn:
            x_tr = train_x.loc[train_index, :]
            for column in categorical_columns:
                x_tr[column] = x_tr[column].astype(int)
            y_tr = train_y[train_index]
            x_te = train_x.loc[test_index, :]
            for column in categorical_columns:
                x_te[column] = x_te[column].astype(int)
        else:
            x_tr = train_x[train_index]
            y_tr = train_y[train_index]
            x_te = train_x[test_index]

        regressor.fit(x_tr, y_tr)

        oof_train[test_index] = regressor.predict(x_te)
        oof_test_skf[i, :] = regressor.predict(test_x)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test