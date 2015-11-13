__author__ = 'Gavin.Chan'

import pandas as pd
import numpy as np
from sklearn import linear_model, cross_validation
from sklearn.metrics import roc_auc_score as auc
from util import set_timer


class SimplePredict:
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
        # Clean target
        for col in self.train_data.columns[self.returns_intraday_index]:
            self.train_data[col] = self.train_data[col].fillna(0)


    def prepare_predictors(self):
        # Start to set run time
        start_time = set_timer()

        # Calculate run time
        set_timer(start_time)

    def predict(self):
        num_of_days = self.test_data.shape[0]
        self.test_prediction = pd.DataFrame(index=range(1, num_of_days + 1),
                                            columns=range(1, 63))
        self.test_prediction = self.test_prediction.fillna(0)

        median = self.train_data.iloc[:,147:209].median(axis = 0)
        median = np.tile(median, (self.test_data.shape[0], 1))
        self.test_prediction.iloc[:,:] = median

    def generate_prediction(self):
        # Start to set run time
        start_time = set_timer()

        # Output the submission to csv
        submission = self.test_prediction.transpose().unstack()
        submission.index = ['_'.join([str(i) for i in s]) for s in submission.index]
        submission.to_csv(self.submission_filename, header=['Predicted'], index_label='Id')

        # Calculate run time
        set_timer(start_time)
