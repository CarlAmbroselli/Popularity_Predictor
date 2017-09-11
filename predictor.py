from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack
# from numpy import hstack
from feature.text_features import TextFeatures
from feature.ngram_features import NGramFeatures
from scipy.sparse import csr_matrix
from feature.word2vec import Word2Vec
from classifier.svr import SVR
from classifier.linear_regression import LinearRegression
# from classifier.naive_bayes import NaiveBayes
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, MaxAbsScaler
from sklearn.metrics import mean_squared_error

class Predictor:
    def fit(self, df):
        '''
        Generate the features from the dataframe and fit the classifiers.
        '''
        feature_matrix = self.calculate_feature_matrix(df)
        print("...using", feature_matrix.shape[1], "features from", ", ".join([feature[0] for feature in self.features]))
        for classifier in self.classifier:
            classifier[1].fit(feature_matrix, self.ground_truth(df))

    def predict(self, df):
        # Generate the features and predict the results.
        feature_matrix = self.calculate_feature_matrix(df)
        predictions = pd.DataFrame()
        for classifier in self.classifier:
            predictions[classifier[0]] = classifier[1].predict(feature_matrix)

        # For each classifier, generate some metrics like recall.
        mean = np.mean(self.ground_truth(df))
        size = self.ground_truth(df).size
        ground_truth = self.ground_truth(df)
        metrics = {
            'dataset': {
                'mean': np.around(mean),
                'size': size,
                'baseline_rmse': np.around(mean_squared_error(ground_truth, np.ones((size, 1)) * mean)**0.5, 2)
            }
        }
        for classifier in self.classifier:
            metrics['prediction_' + classifier[0]] = {
                # 'coef': classifier[1].model.coef_ if classifier[0] == 'linear_regression' else None,
                'rmse': np.around(mean_squared_error(ground_truth, predictions[classifier[0]])**0.5, 2)
            }

        self._metrics = metrics

        return predictions

    def metrics(self):
        return self._metrics

    def ground_truth(self, df):
        return df['comment_count']

    def calculate_feature_matrix(self, df):
        features = [feature[1].extract_features(df) for feature in self.features]
        for feature in features:
            print(feature.shape)
        feature_matrix = hstack(features)
        # feature_matrix = self.features[0][1].extract_features(df)
        scaler = MaxAbsScaler()
        scaled_feature_matrix = scaler.fit_transform(feature_matrix)
        normalize(scaled_feature_matrix, norm='l2', axis=0, copy=False)
        return feature_matrix

    def __init__(self):
        self.features = [
            ('text_features', TextFeatures()),
            ('word2vec', chWord2Vec()),
            # ('ngram_features', NGramFeatures())
        ]

        self.classifier = [
            # ('svr', SVR()),
            ('linear_regression', LinearRegression())
        ]