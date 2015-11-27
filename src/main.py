__author__ = 'Gavin.Chan'

from linear_predict import LinearPredict, FilterLinearPredict
from simple_predict import SimplePredict
from abstract_predict import Predict
from boost_predict import GradientBoostPredict
from analysis import Analysis

def RunAnalysis():
    analysis = Analysis()
    analysis.run_analysis()

def RunForcast():
    # benchmark = Predict()
    # benchmark.run_all()
    #
    # median_predict = SimplePredict(False)
    # median_predict.run_all()
    #
    # mean_predict = SimplePredict(True)
    # mean_predict.run_all()
    #
    # linear_predict = LinearPredict()
    # linear_predict.run_all()
    #
    # filter_linear_predict = FilterLinearPredict(3)
    # filter_linear_predict.run_all()
    #
    # filter_linear_predict = FilterLinearPredict(2)
    # filter_linear_predict.run_all()
    #
    # filter_linear_predict = FilterLinearPredict(1)
    # filter_linear_predict.run_all()
    #
    # filter_linear_predict = FilterLinearPredict(0.5)
    # filter_linear_predict.run_all()

    params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,
              'learning_rate': 0.001, 'max_features': 2, 'loss': 'ls'}
    gradient_boost = GradientBoostPredict(params)
    gradient_boost.run_all()

    params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,
              'learning_rate': 0.001, 'max_features': 4, 'loss': 'ls'}
    gradient_boost = GradientBoostPredict(params)
    gradient_boost.run_all()

    params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,
              'learning_rate': 0.001, 'max_features': 8, 'loss': 'ls'}
    gradient_boost = GradientBoostPredict(params)

    gradient_boost.run_all()
    params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,
              'learning_rate': 0.001, 'loss': 'huber'}
    gradient_boost = GradientBoostPredict(params)
    gradient_boost.run_all()

    params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 1,
              'learning_rate': 0.001, 'loss': 'quantile'}
    gradient_boost = GradientBoostPredict(params)
    gradient_boost.run_all()
if __name__ == '__main__':
    # RunAnalysis()
    RunForcast()
