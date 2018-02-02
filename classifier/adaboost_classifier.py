from sklearn.ensemble import AdaBoostClassifier as Classifier
import numpy as np
import pandas as pd

class AdaboostClassifier:
    def __init__(self):
        self.classifier = Classifier(random_state=42)
        self.model = None

    def fit(self, features, ground_truth):
        self.model = self.classifier.fit(features, ground_truth)

    def predict(self, features):
        assert self.model is not None, 'Executed predict() without calling fit()'
        return self.model.predict(features)