from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd

class LassoRegression:
    def __init__(self):
        self.set_parameter(alpha=1.0)
        self.model = None

    def fit(self, features, ground_truth):
        self.model = self.regressor.fit(features, ground_truth.apply(lambda x: x if x < 100000000 and x > -1000000 else 0))

    def set_parameter(self, alpha=1.0):
        print("Ridge Alpha:", alpha)
        self.regressor = LassoCV()

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)
