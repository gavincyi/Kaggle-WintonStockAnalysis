__author__ = 'Gavin.Chan'

from abstract_predict import Predict
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats


class Analysis(Predict):
    def __init__(self):
        Predict.__init__(self)

    def clean_data(self):
        # Clean features
        median = self.train_data[self.features_index].median(axis=0)
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
                print("Warning occurred in col %s" % col)

            if encoders[col].classes_.shape[0] < 50:
                print("Cleaning col : %s" % col)
                try:
                    self.test_data[col] = encoders[col].transform(self.test_data[col])
                except:
                    print("Test data has different labels with the train data at col %s" % col)

    def describe_out(self):
        stat = pd.DataFrame(columns=['Min', 'Max', 'Mean', 'Median', 'SD', 'Skew', 'Kurt'])

        for i in range(self.returns_prev_days_index[0], self.returns_next_days_index[1]+1):
            data = self.train_data.iloc[:,i]
            n, min_max, mean, var, skew, kurt = stats.describe(data)
            stat.loc[self.train_data.columns[i]] = [min_max[0], min_max[1], mean, data.median(), scipy.sqrt(var), skew, kurt]

        stat.to_csv('../data/stat.csv')


    def describe(self, s):
        n, min_max, mean, var, skew, kurt = stats.describe(s)
        print("Minimun: %.6f" % min_max[0])
        print("Maximum: %.6f" % min_max[1])
        print("Mean: %.6f" % mean)
        print("Standard derivation: %.6f" % scipy.sqrt(var))
        print("Skew: %.6f" % skew)
        print("Kurt: %.6f" % kurt)

    def return_histogram(self, index, diff):
        plt.figure()
        self.train_data.iloc[:, index].hist(bins=20)
        plt.show()

    def run_analysis(self):
        self.get_data()
        self.clean_data()
        self.describe_out()
        # self.describe(self.train_data.iloc[:, self.returns_prev_days_index[0]])
        # self.return_histogram(self.returns_prev_days_index[0], 0.01)
