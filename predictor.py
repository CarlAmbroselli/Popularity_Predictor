import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack as sparse_hstack
from numpy import hstack
from feature.text_features import TextFeatures
from feature.ngram_features import NGramFeatures
from feature.surface_features import SurfaceFeatures
from feature.cumulative_features import CumulativeFeatures
from feature.real_world_features import RealWorldFeatures
from feature.semantic_features import SemanticFeatures
from feature.t_density_features import TDensityFeatures
from feature.subjectivity_features import SubjectivityFeatures
from scipy.sparse import csr_matrix, issparse
from feature.word2vec import Word2Vec
from feature.doc2vec_features import Doc2VecFeatures
from feature.meta_features import MetaFeatures
from classifier.svr import SVR
from classifier.linear_regression import LinearRegression
# from classifier.naive_bayes import NaiveBayes
from classifier.logistic_regression import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


class Predictor:
    def fit(self, df):
        '''
        Generate the features from the dataframe and fit the classifiers.
        '''
        feature_matrix = self.calculate_feature_matrix(df)
        print("...using", feature_matrix.shape[1], "features from", ", ".join([feature[0] for feature in self.features]))
        learners = self.regressors if self._useRegression else self.classifier
        for learner in learners:
            learner[1].fit(feature_matrix, self.ground_truth(df))

    def predict(self, df):
        # Generate the features and predict the results.
        feature_matrix = self.calculate_feature_matrix(df)
        predictions = pd.DataFrame()
        learners = self.regressors if self._useRegression else self.classifier
        for learner in learners:
            predictions[learner[0]] = learner[1].predict(feature_matrix)

        self.regression_metrics(df, predictions) if self._useRegression else self.classifictaion_metrics(df, predictions)

        return predictions

    def classifictaion_metrics(self, df, predictions):
        metrics = {}
        for classifier in self.classifier:
            scores = precision_recall_fscore_support(
                self.ground_truth(df),
                predictions[classifier[0]],
                average='binary'
            )
            metrics[classifier[0]] = dict(zip(['precision', 'recall', 'f-score', 'support'], scores))
            metrics[classifier[0]]['accuracy'] = accuracy_score(self.ground_truth(df), predictions[classifier[0]])

        self._metrics = metrics
        return metrics

    def regression_metrics(self, df, predictions):
        ground_truth = self.ground_truth(df)
        mean = np.mean(ground_truth)
        size = ground_truth.size
        metrics = {
            'dataset': {
                'mean': float("%.2f" % mean),
                'size': size,
                'rmse (avg)': float("%.2f" % mean_squared_error(ground_truth, np.ones((size, 1)) * mean) ** 0.5)
            }
        }
        for learner in self.regressors:
            metrics[learner[0]] = {
                # 'coef': learner[1].model.coef_ if learner[0] == 'linear_regression' else None,
                'rmse': float("%.2f" % mean_squared_error(ground_truth, predictions[learner[0]]) ** 0.5)
            }

        self._metrics = metrics
        return metrics

    def metrics(self):
        return self._metrics

    def set_target(self, column, useRegression):
        self._ground_truth = column
        self._useRegression = useRegression

    def ground_truth(self, df):
        return df[self._ground_truth]

    def calculate_feature_matrix(self, df):
        features = [feature[1].extract_features(df) for feature in self.features]
        has_sparse = False
        for feature in features:
            if issparse(feature):
                has_sparse = True
        if len(features) == 1:
            feature_matrix = features[0]
        else:
            if has_sparse:
                feature_matrix = sparse_hstack(features)
            else:
                feature_matrix = hstack(features)
        # print(feature_matrix)
        scaler = MaxAbsScaler()
        scaled_feature_matrix = scaler.fit_transform(feature_matrix)
        scaled_feature_matrix = normalize(scaled_feature_matrix, norm='l2', axis=0)
        return scaled_feature_matrix

    def __init__(self):
        self.features = [
            ('surface_features', SurfaceFeatures()),
            ('cumulative_features', CumulativeFeatures()),
            ('real_world_features', RealWorldFeatures()),
            ('semantic_features', SemanticFeatures()),
            ('text_features', TextFeatures()),
            ('t_density_features', TDensityFeatures()),
            ('subjectivity_features', SubjectivityFeatures()),
            # ('word2vec', Word2Vec()),
            ('ngram_features', NGramFeatures()),
            ('doc2vec_features', Doc2VecFeatures()),
            ('meta_features', MetaFeatures())
        ]

        self.classifier = [
            ('logistic regression', LogisticRegression()),
        ]

        self.regressors = [
            ('svr', SVR()),
            ('linear_regression', LinearRegression())
        ]