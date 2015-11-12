__author__ = 'Gavin.Chan'

from predict import Predict

if __name__ == '__main__':
    test_predict = Predict()
    test_predict.get_data()
    test_predict.clean_data()
    test_predict.prepare_predictors()
    test_predict.predict()
    test_predict.generate_prediction()