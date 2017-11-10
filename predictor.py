import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack as sparse_hstack, csr_matrix, issparse
from numpy import hstack
from sklearn.preprocessing import normalize, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from classifier.svr import SVR
from classifier.linear_regression import LinearRegression
from classifier.ridge_regression import RidgeRegression
# from classifier.naive_bayes import NaiveBayes
from classifier.logistic_regression import LogisticRegression
import feature as Features
from feature.carl.features import Features as CarlFeatures
from sklearn.metrics import roc_curve, auc
import code

class Predictor:
    def fit(self, df):
        '''
        Generate the features from the dataframe and fit the classifiers.
        '''
        ground_truth = self.ground_truth(df)
        self._training_mean = np.mean(ground_truth)


        feature_matrix = self.calculate_feature_matrix(df)
        print("...using", feature_matrix.shape[1], "features from", ", ".join([feature[0] for feature in self.features]))
        if self._useRegression and len(np.unique(ground_truth)) < 3:
            learners = self.auc_regressors
        else:
            learners = self.regressors if self._useRegression else self.classifier
        for learner in learners:
            learner[1].fit(feature_matrix, self.ground_truth(df))

    def predict(self, df):
        # Generate the features and predict the results.
        feature_matrix = self.calculate_feature_matrix(df)
        predictions = pd.DataFrame()
        if self._useRegression and len(np.unique(self.ground_truth(df))) < 3:
            learners = self.auc_regressors
        else:
            learners = self.regressors if self._useRegression else self.classifier
        for learner in learners:
            predictions[learner[0]] = learner[1].predict(feature_matrix)

        self.regression_metrics(df, predictions) if self._useRegression else self.classification_metrics(df, predictions)
        # self.thresholded_regression_metrics(df, predictions, 0.8) if self._useRegression else self.classification_metrics(df, predictions)

        return predictions

    def thresholded_regression_metrics(self, df, predictions, threshold, metric='precision'):
        metrics = {}
        def normalize(x):
            return x / np.linalg.norm(x)
        for classifier in self.regressors:
            for i in range (1, 100):
                cutoff = i/100
                scores = precision_recall_fscore_support(
                    self.ground_truth(df),
                    normalize(predictions[classifier[0]]) > cutoff,
                    average='binary'
                )
                metrics[classifier[0]] = dict(zip(['precision', 'recall', 'f-score', 'support'], scores))
                metrics[classifier[0]]['accuracy'] = accuracy_score(self.ground_truth(df), normalize(predictions[classifier[0]]) > cutoff)
                metrics[classifier[0]]['accuracy'] = accuracy_score(self.ground_truth(df), normalize(predictions[classifier[0]]) > cutoff)
                if metrics[classifier[0]][metric] > threshold:
                    break
        self._metrics = metrics
        return metrics


    def classification_metrics(self, df, predictions):
        metrics = {}
        for classifier in self.classifier:
            scores = precision_recall_fscore_support(
                self.ground_truth(df),
                predictions[classifier[0]],
                average='binary'
            )
            metrics[classifier[0]] = dict(zip(['precision', 'recall', 'f-score', 'support'], scores))
            # metrics[classifier[0]]['accuracy'] = accuracy_score(self.ground_truth(df), predictions[classifier[0]])
            metrics['class-ratio'] = np.sum(predictions[classifier[0]])/len(predictions[classifier[0]])

        self._metrics = metrics
        return metrics

    def regression_metrics(self, df, predictions):
        ground_truth = self.ground_truth(df)
        mean = np.mean(ground_truth)
        size = ground_truth.size
        metrics = {
            'dataset': {
                'train_mean': float("%.2f" % self.training_mean()),
                'mean': float("%.2f" % mean),
                'size': size,
                'rmse (train mean)': float("%.2f" % mean_squared_error(ground_truth, np.ones((size, 1)) * self.training_mean()) ** 0.5)
            }
        }
        regressors = self.regressors if len(np.unique(ground_truth)) >= 3 else self.auc_regressors
        for learner in regressors:
            if len(np.unique(ground_truth)) < 3:
                false_positive_rate, true_positive_rate, thresholds = roc_curve(ground_truth, predictions[learner[0]])
                roc_auc = auc(false_positive_rate, true_positive_rate)
            else:
                roc_auc = 'n/a'
            metrics[learner[0]] = {
                # 'coef': np.arange(learner[1].model.coef_) if learner[0] == 'linear_regression' else None,
                'rmse': float("%.2f" % mean_squared_error(ground_truth, predictions[learner[0]]) ** 0.5),
                'auc': roc_auc
            }

        self._metrics = metrics
        return metrics

    def metrics(self):
        return self._metrics

    def training_mean(self):
        return self._training_mean

    def set_target(self, column, useRegression):
        self._ground_truth = column
        self._useRegression = useRegression

    def ground_truth(self, df):
        return df[self._ground_truth]

    def calculate_feature_matrix(self, df):
        features = [feature[1].extract_features(df) for feature in self.features]
        print([f.shape for f in features])
        # code.interact(local=locals())
        has_sparse = False
        for feature in features:
            if issparse(feature):
                has_sparse = True
        # [print(f.shape) for f in features]
        if len(features) == 1:
            feature_matrix = features[0]
        else:
            if has_sparse:
                feature_matrix = sparse_hstack(features)
            else:
                feature_matrix = hstack(features)
        scaler = MaxAbsScaler()
        scaled_feature_matrix = scaler.fit_transform(feature_matrix)
        scaled_feature_matrix = normalize(scaled_feature_matrix, norm='l2', axis=0)
        return scaled_feature_matrix

    def __init__(self):
        self.features = [
            # ======== napoles ========
            # ('napoles/bow_features', Features.napoles.BowFeatures()),
            # ('napoles/embeddings_features', Features.napoles.EmbeddingsFeatures()),
            # ('napoles/entity_features', Features.napoles.EntityFeatures()),
            # ('napoles/length_features', Features.napoles.LengthFeatures()),
            # ('napoles/lexicon_features', Features.napoles.LexiconFeatures()),
            # ('napoles/popularity_features', Features.napoles.PopularityFeatures()),
            # ('napoles/pos_features', Features.napoles.POSFeatures()),
            # ('napoles/similarity_features', Features.napoles.SimilarityFeatures()),
            # ('napoles/user_features', Features.napoles.UserFeatures()),

            # ======== tsagkias ========
            ('tsagkias/surface_features', Features.tsagkias.SurfaceFeatures()),
            ('tsagkias/cumulative_features', Features.tsagkias.CumulativeFeatures()),
            ('tsagkias/real_world_features', Features.tsagkias.RealWorldFeatures()),
            # ('tsagkias/semantic_features', Features.tsagkias.SemanticFeatures()),
            ('tsagkias/text_features', Features.tsagkias.TextFeatures()),

            # ======== bandari ========
            # ('bandari/semantic_features', Features.bandari.SemanticFeatures()),
            ('bandari/subjectivity_features', Features.bandari.SubjectivityFeatures()),
            ('bandari/t_density_features', Features.bandari.TDensityFeatures()),

            # ========== own ===========
            # ('subjectivity_features', Features.SubjectivityFeatures()),
            # ('CNN', Features.CNN_Classification()),
            ('word2vec', Features.Word2Vec()),
            ('ngram_features', Features.NGramFeatures()),
            ('doc2vec_features', Features.Doc2VecFeatures()),
            ('meta_features', Features.MetaFeatures()),
            ('topic_features', Features.TopicFeatures()),
            ('semantic_features', Features.SemanticFeatures()),
            ('other_features', CarlFeatures()),
        ]

        self.classifier = [
            ('logistic regression', LogisticRegression()),
        ]

        self.regressors = [
            # ('logistic regression', LogisticRegression()),
            # ('svr', SVR()),
            ('linear_regression', LinearRegression()),
            ('ridge_regression', RidgeRegression()),
        ]        

        self.auc_regressors = [
            ('logistic regression', LogisticRegression(probabilities=True)),
            # ('svr', SVR()),
            # ('linear_regression', LinearRegression()),
            # ('ridge_regression', RidgeRegression()),
        ]
