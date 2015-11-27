__author__ = 'Gavin.Chan'

import pandas as pd
from linear_predict import SklearnPredict
from sklearn import ensemble, cross_validation


class GradientBoostPredict(SklearnPredict):
    def __init__(self, params):
        SklearnPredict.__init__(self)
        self.params = params

    def fit(self, X, y):
        predictor = []

        kf = cross_validation.KFold(y.shape[0], n_folds=3, random_state=1)
        for itr, icv in kf:
            clf = ensemble.GradientBoostingRegressor(**(self.params))
            clf.fit(X.iloc[itr], y.iloc[itr].values)
            predictor.append(clf)

        return predictor
