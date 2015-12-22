__author__ = 'Gavin.Chan'

from linear_predict import SklearnPredict
from sklearn import cross_validation
import xgboost as xgb
import pandas as pd

class XGBoostPredict(SklearnPredict):
    def __init__(self, param):
        SklearnPredict.__init__(self)
        self.param = param

    def fit(self, X, y):
        predictor = []
        kf = cross_validation.KFold(y.shape[0], n_folds = 3, random_state = 1)

        for itr, icv in kf:
            dtrain = xgb.DMatrix(X.iloc[itr].as_matrix(), label=y.iloc[itr].as_matrix())
            predictor.append(xgb.train(params=self.param, dtrain=dtrain, num_boost_round=1))

        return predictor

    def evaluate_unbatch_error(self, X):
        X_matrix = xgb.DMatrix(X.as_matrix())
        train_unbatch_predict = pd.DataFrame(index=range(1, X.shape[0] + 1),
                                             columns=range(1, 63))
        train_unbatch_predict = train_unbatch_predict.fillna(0)

        prediction_index = range(61,63)
        for i in range(0, len(prediction_index)):
            y_predict = pd.DataFrame()

            for index, xgb_model in enumerate(self.predictors[i]):
                y_predict[index] = xgb_model.predict(X_matrix)

            y_final = y_predict.mean(axis = 1)
            train_unbatch_predict[prediction_index[i]] = pd.Series(y_final).values

        return train_unbatch_predict


