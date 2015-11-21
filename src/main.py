__author__ = 'Gavin.Chan'

from linear_predict import LinearPredict
from simple_predict import SimplePredict
from abstract_predict import Predict, PredictTest

if __name__ == '__main__':
    benchmark = Predict()
    benchmark.run_all(False)

    median_predict = SimplePredict()
    median_predict.run_all(False)

    linear_predict = LinearPredict()
    linear_predict.run_all(False)
