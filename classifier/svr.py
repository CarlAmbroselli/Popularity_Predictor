from sklearn.svm import SVR as sklearn_svr
import numpy as np
import pandas as pd

class SVR:
    def __init__(self):
        self.svr = sklearn_svr(kernel='linear')
        self.model = None

    def fit(self, features, ground_truth):
        self.model = self.svr.fit(features, ground_truth)

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)