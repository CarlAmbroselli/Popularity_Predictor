from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier
import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, probabilities=False):
        self.lr = LogisticRegressionClassifier(random_state=0, class_weight='balanced')
        self.model = None
        self.probabilities = probabilities

    def fit(self, features, ground_truth):
        self.model = self.lr.fit(features, ground_truth)

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        if self.probabilities:
            return self.model.predict_proba(features).T[1].T
        else:
            return self.model.predict(features)
