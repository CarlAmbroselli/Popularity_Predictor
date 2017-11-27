from sklearn.linear_model import Ridge as RR
import numpy as np
import pandas as pd

class RidgeRegression:
    def __init__(self):
        self.regressor = RR() # RR(normalize=True)
        self.model = None

    def fit(self, features, ground_truth):
        self.model = self.regressor.fit(features, ground_truth)

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)