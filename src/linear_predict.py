__author__ = 'Gavin.Chan'

import pandas as pd
import math
from abstract_predict import Predict
from sklearn import linear_model, cross_validation
from scipy import stats

class SklearnPredict(Predict):
    def __init__(self):
        Predict.__init__(self)

    def batch_filter(self, X, Y):
        return X, Y

    def fit(self, X, y):
        pass

    def prepare_predictors(self):
        X_batch = self.train_data.iloc[self.train_batch_index, self.features_filtered_index]
        X_unbatch = self.train_data.iloc[self.train_unbatch_index, self.features_filtered_index]
        Y = self.train_data.iloc[self.train_batch_index, self.returns_next_days_index]

        for col in Y.columns:
            x, y = self.batch_filter(X_batch, Y[col])
            predictor = self.fit(x, y)
            self.predictors[col] = predictor

        train_unbatch_predict = pd.DataFrame(index=self.train_unbatch_index,
                                             columns=range(1, 63))
        train_unbatch_predict = train_unbatch_predict.fillna(0)

        prediction_index = range(61,63)
        for i in range(0, Y.shape[1]):
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

class LinearPredict(SklearnPredict):
    def __init__(self):
        SklearnPredict.__init__(self)

    def fit(self, X, y):
        predictor = []

        kf = cross_validation.KFold(y.shape[0], n_folds = 3, random_state = 1)
        for itr, icv in kf:
            clf = linear_model.LinearRegression()
            clf.fit(X.iloc[itr], y.iloc[itr].values)
            predictor.append(clf)

        return predictor


class FilterLinearPredict(LinearPredict):
    def __init__(self, range):
        LinearPredict.__init__(self)
        self.range = range

    def batch_filter(self, X, Y):
        n, min_max, mean, var, skew, kurt = stats.describe(Y)
        sd = math.sqrt(var)
        y_index = Y[(Y > mean - self.range * sd).values & (Y < mean + self.range * sd).values].index.tolist()
        X = X.iloc[y_index, :]
        Y = Y[y_index]
        return X, Y
