import xgboost as xgb
import numpy as np


class skxgboost:
    def __init__(self, params, boost_rounds):
        self.params = params
        self.boost_rounds = boost_rounds
        self.clf = 0

    def fit(self, x, y):
        y[y == -1] = 0
        self.clf = xgb.train(self.params,
                             xgb.DMatrix(x, y),
                             num_boost_round=self.boost_rounds,
                             verbose_eval=False,
                             maximize=False)
        return self

    def predict_proba(self, x):
        prob = np.array(self.clf.predict(xgb.DMatrix(x),
                                         ntree_limit=self.clf.best_iteration))
        return np.array([1-prob, prob]).transpose()
