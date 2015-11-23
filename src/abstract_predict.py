__author__ = 'Gavin.Chan'

import random
import pandas as pd
import numpy as np
import time

from sklearn import preprocessing

LOG_MODE = 0
TEST = 0

class BasePredict:
    def __init__(self):
        random.seed(123)
        self.train_data_filename = ''
        self.test_data_filename = ''
        self.submission_filename = ''
        self.features_index = range(1, 26)
        self.features_filtered_index = [3,5,6,7,8,9,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27]
        self.returns_prev_days_index = [26, 27]
        self.returns_intraday_index = range(28, 207)
        self.returns_predict_index = range(147, 209)
        self.returns_next_days_index = [207, 208]
        self.weight_intraday_index = 209
        self.weight_daily_index = 210
        self.train_batch_index = []
        self.train_unbatch_index = []

        # Data
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.test_prediction = pd.DataFrame()
        self.predictor = []

    @staticmethod
    def run(f):
        start = time.localtime()
        if LOG_MODE > 0:
            print("===================================================")
            print("Start running function %s " % f.__name__)
            print("Start time = %s)" % time.strftime("%c", start))

        f()

        end = time.localtime()
        if LOG_MODE > 0:
            print("Finished running function %s" % f.__name__)
            print("Run time = %s" % (time.mktime(end) - time.mktime(start)))

    def get_data(self):
        self.train_data = pd.read_csv(self.train_data_filename)
        self.test_data = pd.read_csv(self.test_data_filename)

        # Prepare batch and unbatch index
        count = self.train_data.shape[0]
        self.train_batch_index = range(0, int(count*2/3))
        self.train_unbatch_index = range(int(count*2/3), count)

        # Prepare test prediction
        num_of_days = self.test_data.shape[0]
        self.test_prediction = pd.DataFrame(index=range(1, num_of_days + 1),
                                            columns=range(1, 63))
        self.test_prediction = self.test_prediction.fillna(0)

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
                try:
                    self.test_data[col] = encoders[col].transform(self.test_data[col])
                except:
                    print("Test data has different labels with the train data at col %s" % col)

    def prepare_predictors(self):
        # Predict unbatch prediction
        train_unbatch_predict = pd.DataFrame(index=self.train_unbatch_index,
                                             columns=range(1, 63))
        train_unbatch_predict = train_unbatch_predict.fillna(0)

        error = self.evaluate_error(self.train_data.iloc[self.train_unbatch_index,:],
                                    train_unbatch_predict)
        print("%s : Unbatched error = %.4f" % (self.__class__.__name__, error))

    def predict(self):
        pass

    def generate_prediction(self):
        # Output the submission to csv
        submission = self.test_prediction.transpose().unstack()
        submission.index = ['_'.join([str(i) for i in s]) for s in submission.index]
        submission.to_csv(self.submission_filename, header=['Predicted'], index_label='Id')

    def evaluate_error(self, actual, predict):
        weight = pd.DataFrame(index=range(0, actual.shape[0]),
                                    columns=range(1, 63))
        abs_error = abs(actual.iloc[:,self.returns_predict_index].values - predict.values)
        weight.iloc[:,0:2]   = np.tile(actual.iloc[:,self.weight_daily_index], (2, 1)).T
        weight.iloc[:,2:60]  = np.tile(actual.iloc[:,self.weight_intraday_index], (58, 1)).T
        weight.iloc[:,60:62] = np.tile(actual.iloc[:,self.weight_daily_index], (2, 1)).T
        abs_error = abs_error * weight.values
        count = abs_error.shape[0] * abs_error.shape[1]
        return abs_error.sum()/count

    def run_all(self, is_predict):
        self.run(self.get_data)
        self.run(self.clean_data)
        self.run(self.prepare_predictors)

        if is_predict:
            self.run(self.predict)
            self.run(self.generate_prediction)


class PredictSubmit(BasePredict):
    def __init__(self):
        BasePredict.__init__(self)
        self.train_data_filename = '../data/train.csv'
        self.test_data_filename = '../data/test.csv'
        self.submission_filename = '../data/submission.csv'


class PredictTest(BasePredict):
    def __init__(self):
        BasePredict.__init__(self)
        self.train_data_filename = '../data/train_trim.csv'
        self.test_data_filename = '../data/test_trim.csv'
        self.submission_filename = '../data/submission_trim.csv'


class Predict(PredictSubmit, PredictTest):
    def __init__(self):
        if TEST == 1:
            PredictTest.__init__(self)
        else:
            PredictSubmit.__init__(self)
