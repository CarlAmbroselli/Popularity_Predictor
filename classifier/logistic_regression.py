from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier
import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self):
        self.lr = LogisticRegressionClassifier(random_state=0, class_weight='balanced')
        self.model = None

    def fit(self, features, ground_truth):
        self.model = self.lr.fit(features, ground_truth)

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)