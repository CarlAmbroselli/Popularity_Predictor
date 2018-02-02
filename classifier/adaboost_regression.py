from sklearn.ensemble import AdaBoostRegressor as Regressor
import numpy as np
import pandas as pd

class AdaboostRegression:
    def __init__(self):
        self.regressor = Regressor(random_state=42)
        self.model = None

    def fit(self, features, ground_truth):
        self.model = self.regressor.fit(features, ground_truth)

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)