from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

class XGBoostRegression:
    def __init__(self):
        self.regressor = XGBRegressor() #random_state=0, class_weight='balanced')
        self.model = None

    def fit(self, features, ground_truth):
        self.model = self.regressor.fit(csc_matrix(features, ground_truth))

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(csc_matrix(features))

