__author__ = 'Gavin.Chan'

from abstract_predict import Predict, PredictTest
import pandas as pd
import numpy as np
from sklearn import linear_model, cross_validation, preprocessing


class LinearPredict(Predict):
    def clean_data(self):
        # Clean features
        median = self.train_data[self.features_index].median(axis = 0)
        for col in self.train_data.columns[self.features_index]:
            self.train_data[col] = self.train_data[col].fillna(median[col])
            self.test_data[col] = self.test_data[col].fillna(median[col])

        # Clean target
        for col in self.train_data.columns[self.returns_intraday_index]:
            self.train_data[col] = self.train_data[col].fillna(0)

        encoders = {}
        for col in self.train_data.columns[1:self.returns_prev_days_index[0]]:
            encoders[col] = preprocessing.LabelEncoder()
            try:
                self.train_data[col] = encoders[col].fit_transform(self.train_data[col])
            except:
                print("Warning occured in col %s" % col)

            if encoders[col].classes_.shape[0] < 50:
                print("Cleaning col : %s" % col)
                try:
                    self.test_data[col] = encoders[col].transform(self.test_data[col])
                except:
                    print("Test data has different labels with the train data at col %s" % col)

    def linear_predict(self, X, y):
        predictor = []

        kf = cross_validation.KFold(y.shape[0], n_folds = 3, random_state = 1)
        for i in range(0, y.shape[1]):
            predictor.append([])
            for itr, icv in kf:
                clf = linear_model.LinearRegression()
                clf.fit(X.iloc[itr], y.iloc[itr, i])
                predictor[i].append(clf)

        predictor = pd.Series(predictor, index=y.columns)
        return predictor

    def prepare_predictors(self):
        X = self.train_data.iloc[self.train_batch_index, self.features_filtered_index]
        X_unbatch = self.train_data.iloc[self.train_unbatch_index, self.features_filtered_index]
        y = self.train_data.iloc[self.train_batch_index, self.returns_next_days_index]
        predictor = self.linear_predict(X, y)
        train_unbatch_predict = pd.DataFrame(index=self.train_unbatch_index,
                                             columns=range(1, 63))
        train_unbatch_predict = train_unbatch_predict.fillna(0)

        prediction_index = [61, 62]
        for i in range(0, y.shape[1]):
            y_predict = pd.DataFrame()
            for index, clf in enumerate(predictor[i]):
                y_predict[index] = clf.predict(X_unbatch)

            y_final = y_predict.mean(axis=1)
            train_unbatch_predict[prediction_index[i]] = pd.Series(y_final).values

        error = self.evaluate_error(self.train_data.iloc[self.train_unbatch_index,:],
                                    train_unbatch_predict)
        print("%s : Unbatched error = %.4f" % (self.__class__.__name__, error))

    def predict(self):
        weight = self.train_data[self.train_data.columns[self.weight_intraday_index]]
        X = self.train_data[self.features_filtered_index]
        y = self.train_data.iloc[:,self.returns_next_days_index]
        errors = []
        kf = cross_validation.KFold(y.shape[0], n_folds = 3, random_state = 1)

        for i in range(0, 2):
            self.predictor.append([])

            for itr, icv in kf:
                clf = linear_model.LinearRegression()
                count = len(icv)
                clf.fit(X.iloc[itr], y.iloc[itr, i])
                self.predictor[i].append(clf)

                # y_predict = clf.predict(X.iloc[icv])
                # error = sum(abs(y_predict - y.iloc[icv])*weight.iloc[icv])/count
                # errors.append(error)

        X = self.test_data[self.features_filtered_index]
        y = pd.DataFrame()

        prediction_index = [61, 62]
        for i in range(0,2):
            for index, clf in enumerate(self.predictor[i]):
                y[index] = clf.predict(X)

            y_final = y.mean(axis=1)
            self.test_prediction[prediction_index[i]] = pd.Series(y_final).values

