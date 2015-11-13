__author__ = 'Gavin.Chan'

from linear_predict import LinearPredict
from simple_predict import SimplePredict

if __name__ == '__main__':
    test_predict = SimplePredict()
    test_predict.get_data()
    test_predict.clean_data()
    test_predict.prepare_predictors()
    test_predict.predict()
    test_predict.generate_prediction()