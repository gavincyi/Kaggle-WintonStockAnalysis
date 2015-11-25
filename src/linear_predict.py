__author__ = 'Gavin.Chan'

from abstract_predict import Predict, PredictTest
import pandas as pd
import numpy as np
import math
from sklearn import linear_model, cross_validation
from scipy import stats


class LinearPredict(Predict):
    def __init__(self):
        Predict.__init__(self)

    def linear_predict(self, X, y):
        predictor = []

        kf = cross_validation.KFold(y.shape[0], n_folds = 3, random_state = 1)
        for itr, icv in kf:
            clf = linear_model.LinearRegression()
            clf.fit(X.iloc[itr], y.iloc[itr])
            predictor.append(clf)

        return predictor

    def prepare_predictors(self):
        X = self.train_data.iloc[self.train_batch_index, self.features_filtered_index]
        X_unbatch = self.train_data.iloc[self.train_unbatch_index, self.features_filtered_index]
        y = self.train_data.iloc[self.train_batch_index, self.returns_next_days_index]

        for col in y.columns:
            predictor = self.linear_predict(X, y[col])
            self.predictors[col] = predictor

        train_unbatch_predict = pd.DataFrame(index=self.train_unbatch_index,
                                             columns=range(1, 63))
        train_unbatch_predict = train_unbatch_predict.fillna(0)

        prediction_index = range(61,63)
        for i in range(0, y.shape[1]):
            y_predict = pd.DataFrame()
            for index, clf in enumerate(self.predictors[i]):
                y_predict[index] = clf.predict(X_unbatch)

            y_final = y_predict.mean(axis=1)
            train_unbatch_predict[prediction_index[i]] = pd.Series(y_final).values

        error = self.evaluate_error(self.train_data.iloc[self.train_unbatch_index,:],
                                    train_unbatch_predict)
        print("%s : Unbatched error = %.4f" % (self.__class__.__name__, error))

    def predict(self):
        X = self.test_data[self.features_filtered_index]

        prediction_index = range(61,63)
        for i in range(0, self.predictors.shape[0]):
            y_predict = pd.DataFrame()
            for index, clf in enumerate(self.predictors[i]):
                y_predict[index] = clf.predict(X)

            y_final = y_predict.mean(axis=1)
            self.test_prediction[prediction_index[i]] = pd.Series(y_final).values

        mean_train = self.train_data.iloc[:, self.returns_next_days_index].mean(axis = 0)
        mean_test = self.test_prediction[prediction_index].mean(axis = 0)
        print("mean_train = \n%s" % mean_train)
        print("mean_test = \n%s" % mean_test)


class FilterLinearPredict(LinearPredict):
    def __init__(self, range):
        LinearPredict.__init__(self)
        self.range = range

    def prepare_predictors(self):
        X = self.train_data.iloc[self.train_batch_index, self.features_filtered_index]
        X_unbatch = self.train_data.iloc[self.train_unbatch_index, self.features_filtered_index]
        y = self.train_data.iloc[self.train_batch_index, self.returns_next_days_index]

        for col in y.columns:
            y_data = y[col]
            n, min_max, mean, var, skew, kurt = stats.describe(y_data)
            sd = math.sqrt(var)
            y_index = y_data[(y_data > mean - self.range * sd).values & (y_data < mean + self.range * sd).values].index.tolist()
            predictor = self.linear_predict(X.iloc[y_index, :], pd.Series(y_data[y_index]))
            self.predictors[col] = predictor

        train_unbatch_predict = pd.DataFrame(index=self.train_unbatch_index,
                                             columns=range(1, 63))
        train_unbatch_predict = train_unbatch_predict.fillna(0)

        prediction_index = range(61, 63)
        for i in range(0, y.shape[1]):
            y_predict = pd.DataFrame()
            for index, clf in enumerate(self.predictors[i]):
                y_predict[index] = clf.predict(X_unbatch)

            y_final = y_predict.mean(axis=1)
            train_unbatch_predict[prediction_index[i]] = pd.Series(y_final).values

        error = self.evaluate_error(self.train_data.iloc[self.train_unbatch_index,:],
                                    train_unbatch_predict)
        print("%s : Unbatched error = %.4f" % (self.__class__.__name__, error))

    def predict(self):
        LinearPredict.predict(self)
