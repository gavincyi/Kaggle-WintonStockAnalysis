__author__ = 'Gavin.Chan'

import pandas as pd
import numpy as np
from sklearn import linear_model, cross_validation
from sklearn.metrics import roc_auc_score as auc
from util import set_timer


class Predict:
    def __init__(self):
        # Constants
        self.train_data_filename = '../data/train.csv'
        self.test_data_filename = '../data/test.csv'
        self.submission_filename = '../data/submission.csv'
        self.features_index = range(1, 26)
        self.features_filtered_index = [3,5,6,7,8,9,11,12,13,14,15,16,17,18,19,22,23,24,25]
        self.returns_prev_days_index = [26, 27]
        self.returns_intraday_index = range(28, 207)
        self.returns_next_days_index = [207, 208]
        self.weight_intraday_index = 209
        self.weight_daily_index = 210

        # Data
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.test_prediction = pd.DataFrame()
        self.predictor = []

    def get_data(self):
        # Start to set run time
        start_time = set_timer()

        self.train_data = pd.read_csv(self.train_data_filename)
        self.test_data = pd.read_csv(self.test_data_filename)

        print("Features count = %d" % len(self.features_filtered_index))

        # Calculate run time
        set_timer(start_time)

    def clean_data(self):
        # Clean features
        mean = self.train_data[self.features_index].sum()/self.train_data.shape[0]
        for col in self.train_data.columns[self.features_index]:
            self.train_data[col] = self.train_data[col].fillna(mean[col])
            self.test_data[col] = self.test_data[col].fillna(mean[col])

        # Clean target
        for col in self.train_data.columns[self.returns_intraday_index]:
            self.train_data[col] = self.train_data[col].fillna(0)


    def prepare_predictors(self):
        # Start to set run time
        start_time = set_timer()

        weight = self.train_data[self.train_data.columns[self.weight_intraday_index]]
        X = self.train_data[self.features_filtered_index]
        y = self.train_data[self.train_data.columns[self.returns_intraday_index[0]]]
        errors = []
        kf = cross_validation.KFold(y.shape[0], n_folds = 3, random_state = 1)
        clf = linear_model.LinearRegression()

        for itr, icv in kf:
            count = len(icv)
            clf.fit(X.iloc[itr], y.iloc[itr])
            self.predictor.append(clf)
            # y_predict = clf.predict(X.iloc[icv])
            # error = sum(abs(y_predict - y.iloc[icv])*weight.iloc[icv])/count
            # errors.append(error)

        # Calculate run time
        set_timer(start_time)


    def cross_validate(self):
        print('CV running...')

    def predict(self):
        num_of_days = self.test_data.shape[0]
        self.test_prediction = pd.DataFrame(index=range(1, num_of_days + 1),
                                            columns=range(1, 63))
        self.test_prediction = self.test_prediction.fillna(0)

        X = self.test_data[self.features_filtered_index]
        y = pd.DataFrame()
        for index, clf in enumerate(self.predictor):
            y[index] = clf.predict(X)

        y_final = y.mean(axis=1)
        self.test_prediction[1] = pd.Series(y_final).values


    def generate_prediction(self):
        # Start to set run time
        start_time = set_timer()

        # Output the submission to csv
        submission = self.test_prediction.transpose().unstack()
        submission.index = ['_'.join([str(i) for i in s]) for s in submission.index]
        submission.to_csv(self.submission_filename, header=['Predicted'], index_label='Id')

        # Calculate run time
        set_timer(start_time)
