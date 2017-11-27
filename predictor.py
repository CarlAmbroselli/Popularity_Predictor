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
from feature.carl.zeit_features import ZeitFeatures#
from sklearn.metrics import roc_curve, auc
import code
from feature.carl.after_publication_features import AfterPublicationFeatures
import pickle

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
            
            # code.interact(local=locals())
            metrics[learner[0]] = {
                # 'coef': np.arange(learner[1].model.coef_), #if learner[0] == 'linear_regression' else None,
                'rmse': float("%.2f" % mean_squared_error(ground_truth, predictions[learner[0]]) ** 0.5),
                'auc': roc_auc
            }
            try:
                print('')
                # pickle.dump(learner[1].model.coef_, open('coef_.pickle', "wb"), protocol=4)
            except Exception as e:
                print(e)

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
        # print([f.shape for f in features])
        # code.interact(local=locals())
        has_sparse = False
        for feature in features:
            if issparse(feature):
                has_sparse = True
        [print(f.shape) for f in features]
        if len(features) == 1:
            feature_matrix = features[0]
        else:
            if has_sparse:
                feature_matrix = sparse_hstack(features)
            else:
                feature_matrix = hstack(features)
        # scaler = MaxAbsScaler()
        # scaled_feature_matrix = scaler.fit_transform(feature_matrix)
        # scaled_feature_matrix = normalize(scaled_feature_matrix, norm='l2', axis=0)
        # return scaled_feature_matrix
        
        # code.interact(local=locals())
        return feature_matrix

    def __init__(self):
        self.always_use_these_features  = [
           #  ('tsagkias/surface_features', Features.tsagkias.SurfaceFeatures()),
           #  ('tsagkias/cumulative_features', Features.tsagkias.CumulativeFeatures()),
           #  ('tsagkias/real_world_features', Features.tsagkias.RealWorldFeatures()),
           #  ('tsagkias/text_features', Features.tsagkias.TextFeatures()),
           #  ('bandari/subjectivity_features', Features.bandari.SubjectivityFeatures()),
           #  ('bandari/t_density_features', Features.bandari.TDensityFeatures()),
           #  ('word2vec-100', Features.Word2Vec(num_dimensions=100)),
           #  ('ngram_features-(1,3)', Features.NGramFeatures((1, 3))),
           #  ('doc2vec_features-100', Features.Doc2VecFeatures(num_dimensions=100)),
           #  ('meta_features', Features.MetaFeatures()),
           #  ('topic_features-100', Features.TopicFeatures(num_topics=100)),
           #  ('semantic_features', Features.SemanticFeatures()),
           #  ('other_features', CarlFeatures()),
        ]

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

            # # ======== tsagkias ========
            # ('tsagkias/surface_features', Features.tsagkias.SurfaceFeatures()),
            # ('tsagkias/cumulative_features', Features.tsagkias.CumulativeFeatures()),
            # ('tsagkias/real_world_features', Features.tsagkias.RealWorldFeatures()),
            # ('tsagkias/semantic_features', Features.tsagkias.SemanticFeatures()),
            # ('tsagkias/text_features', Features.tsagkias.TextFeatures()),

            # # ======== bandari ========
            # ('bandari/semantic_features', Features.bandari.SemanticFeatures()),
            # ('bandari/subjectivity_features', Features.bandari.SubjectivityFeatures()),
            # ('bandari/t_density_features', Features.bandari.TDensityFeatures()),

            # # ========== own ===========
            # # ('subjectivity_features', Features.SubjectivityFeatures()),
            # # ('CNN', Features.CNN_Classification()),
            # ('word2vec-50', Features.Word2Vec(num_dimensions=50)),
            # ('word2vec-100/2', Features.Word2Vec(num_dimensions=100, window_size=2)),
            # ('word2vec-100/3', Features.Word2Vec(num_dimensions=100, window_size=3)),
            # ('word2vec-100/4', Features.Word2Vec(num_dimensions=100, window_size=4)),
            # ('word2vec-100/5', Features.Word2Vec(num_dimensions=100, window_size=5)),
            # ('word2vec-100/6', Features.Word2Vec(num_dimensions=100, window_size=6)),
            # ('word2vec-100/7', Features.Word2Vec(num_dimensions=100, window_size=7)),
            # ('word2vec-100/8', Features.Word2Vec(num_dimensions=100, window_size=8)),
            # ('word2vec-100', Features.Word2Vec(num_dimensions=100)),
            # ('word2vec-150', Features.Word2Vec(num_dimensions=150)),
            # ('word2vec-200', Features.Word2Vec(num_dimensions=200)),
            # ('word2vec-250', Features.Word2Vec(num_dimensions=250)),
            # ('word2vec-500', Features.Word2Vec(num_dimensions=500)),
            # ('stemmed_headline_min-2_ngram_features-(1)', Features.NGramHeadlineFeatures((1,1))),
            # ('stemmed_headline_min-2_ngram_features-(1,2)', Features.NGramHeadlineFeatures((1,2))),
            # ('stemmed_headline_min-2_ngram_features-(1,3)', Features.NGramHeadlineFeatures((1,3))),
            # ('stemmed_headline_min-2_ngram_features-(2)', Features.NGramHeadlineFeatures((2,2))),
            # ('stemmed_headline_min-2_ngram_features-(2,3)', Features.NGramHeadlineFeatures((2,3))),
            # ('stemmed_headline_min-2_ngram_features-(3)', Features.NGramHeadlineFeatures((3,3))),
            # ('min-2_ngram_features-(1)', Features.NGramFeatures((1,1))),
            # ('min-2_ngram_features-(1,2)', Features.NGramFeatures((1,2))),
            # ('min-2_ngram_features-(1,3)', Features.NGramFeatures((1,3))),
            # ('min-2_ngram_features-(2)', Features.NGramFeatures((2,2))),
            # ('min-2_ngram_features-(2,3)', Features.NGramFeatures((2,3))),
            # ('min-2_ngram_features-(3)', Features.NGramFeatures((3,3))),
            # ('doc2vec_features-50', Features.Doc2VecFeatures(num_dimensions=50)),
            # ('doc2vec_features-100', Features.Doc2VecFeatures(num_dimensions=100)),
            # ('doc2vec_features-1/2', Features.Doc2VecFeatures(num_dimensions=1, window_size=2)),
            # ('doc2vec_features-5/2', Features.Doc2VecFeatures(num_dimensions=5, window_size=2)),
            # ('doc2vec_features-10/2', Features.Doc2VecFeatures(num_dimensions=10, window_size=2)),
            # ('doc2vec_features-25/2', Features.Doc2VecFeatures(num_dimensions=25, window_size=2)),
            # ('doc2vec_features-50/2', Features.Doc2VecFeatures(num_dimensions=50, window_size=2)),
            # ('doc2vec_features-100/2', Features.Doc2VecFeatures(num_dimensions=100, window_size=2)), # BEST!!!
            # ('doc2vec_features-100/3', Features.Doc2VecFeatures(num_dimensions=100, window_size=3)),
            # ('doc2vec_features-100/4', Features.Doc2VecFeatures(num_dimensions=100, window_size=4)),
            # ('doc2vec_features-100/5', Features.Doc2VecFeatures(num_dimensions=100, window_size=5)),
            # ('doc2vec_features-100/6', Features.Doc2VecFeatures(num_dimensions=100, window_size=6)),
            # ('doc2vec_features-100/7', Features.Doc2VecFeatures(num_dimensions=100, window_size=7)),
            # ('doc2vec_features-100/8', Features.Doc2VecFeatures(num_dimensions=100, window_size=8)),
            # ('doc2vec_features-150', Features.Doc2VecFeatures(num_dimensions=150)),
            # ('doc2vec_features-200', Features.Doc2VecFeatures(num_dimensions=150)),
            # ('doc2vec_features-250', Features.Doc2VecFeatures(num_dimensions=250)),
            # ('doc2vec_features-500', Features.Doc2VecFeatures(num_dimensions=500)),
            # ('meta_features', Features.MetaFeatures()),
            # ('topic_features-50', Features.TopicFeatures(num_topics=50)),
            # ('topic_features-100', Features.TopicFeatures(num_topics=100)),
            # ('topic_features-150', Features.TopicFeatures(num_topics=150)),
            # ('topic_features-200', Features.TopicFeatures(num_topics=200)),
            # ('topic_features-250', Features.TopicFeatures(num_topics=250)),
            # ('topic_features-500', Features.TopicFeatures(num_topics=500)),
            ('topic_features-750', Features.TopicFeatures(num_topics=750)),
            ('topic_features-1000', Features.TopicFeatures(num_topics=1000)),
            # ('named_entities_features', Features.SemanticFeatures()),
            # ('other_features', CarlFeatures()),
            # ('zeit_features_shared_on_facebook', ZeitFeatures()),
            # ('keyword_features', Features.KeywordFeatures()),

            # ('after_publication_features-2', AfterPublicationFeatures(maximum_time=2)),
            # ('after_publication_features-4', AfterPublicationFeatures(maximum_time=4)),
            # ('after_publication_features-8', AfterPublicationFeatures(maximum_time=8)),
            # ('after_publication_features-16', AfterPublicationFeatures(maximum_time=16)),
            # ('after_publication_features-32', AfterPublicationFeatures(maximum_time=32)),
            # ('after_publication_features-64', AfterPublicationFeatures(maximum_time=64)),
            # ('after_publication_features-128', AfterPublicationFeatures(maximum_time=128)),
            # ('after_publication_features-256', AfterPublicationFeatures(maximum_time=256)),
            # ('after_publication_features-512', AfterPublicationFeatures(maximum_time=512)),
            # ('after_publication_features-1024', AfterPublicationFeatures(maximum_time=1024)),
            # ('after_publication_features-2048', AfterPublicationFeatures(maximum_time=2048)),
            # ('after_publication_features-4096', AfterPublicationFeatures(maximum_time=4096)),
            # ('after_publication_features-8192', AfterPublicationFeatures(maximum_time=8192)),
            # ('after_publication_features-16384', AfterPublicationFeatures(maximum_time=16384)),
            # ('after_publication_features-32768', AfterPublicationFeatures(maximum_time=32768)),
            # ('after_publication_features-65536', AfterPublicationFeatures(maximum_time=65536)),
            # ('after_publication_features-131072', AfterPublicationFeatures(maximum_time=131072)),
            # ('after_publication_features-262144', AfterPublicationFeatures(maximum_time=262144)),
            # ('after_publication_features-524288', AfterPublicationFeatures(maximum_time=524288)),
            # ('after_publication_features-1048576', AfterPublicationFeatures(maximum_time=1048576)),
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
