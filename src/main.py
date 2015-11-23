__author__ = 'Gavin.Chan'

from linear_predict import LinearPredict, FilterLinearPredict
from simple_predict import SimplePredict
from abstract_predict import Predict
from analysis import Analysis

def RunAnalysis():
    analysis = Analysis()
    analysis.run_analysis()

def RunForcast():
    benchmark = Predict()
    benchmark.run_all(False)

    median_predict = SimplePredict()
    median_predict.run_all(False)

    linear_predict = LinearPredict()
    linear_predict.run_all(False)

    filter_linear_predict = FilterLinearPredict()
    filter_linear_predict.run_all(False)

if __name__ == '__main__':
    RunForcast()
