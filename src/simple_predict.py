__author__ = 'Gavin.Chan'

from abstract_predict import Predict, PredictTest
import pandas as pd
import numpy as np
from util import set_timer


class SimplePredict(Predict):
    def __init__(self, is_mean):
        Predict.__init__(self)
        if is_mean:
            self.is_mean = True
        else:
            self.is_mean = False


    def prepare_predictors(self):
        # Predict unbatch prediction
        estimator = self.train_data.iloc[self.train_batch_index, self.returns_predict_index]
        if self.is_mean:
            estimator = estimator.mean(axis = 0)
        else:
            estimator = estimator.median(axis = 0)

        estimator = np.tile(estimator, (len(self.train_unbatch_index), 1))

        train_unbatch_predict = pd.DataFrame(index=self.train_unbatch_index,
                                            columns=range(1, 63))
        train_unbatch_predict= train_unbatch_predict.fillna(0)
        train_unbatch_predict.iloc[:,] = estimator

        error = self.evaluate_error(self.train_data.iloc[self.train_unbatch_index,:],
                                    train_unbatch_predict)
        print("%s : Unbatched error = %.4f" % (self.__class__.__name__, error))


    def predict(self):
        median = self.train_data.iloc[:,147:209].median(axis = 0)
        median = np.tile(median, (self.test_data.shape[0], 1))
        self.test_prediction.iloc[:,] = median

