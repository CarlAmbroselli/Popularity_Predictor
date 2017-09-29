from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from feature.features import Features

class NGramFeatures(Features):
  def __init__(self):
    super().__init__('ngram_features')
    self.first_run = True

  def _extract_features(self, df):
    features = None
    if self.first_run:
      self.count_vectorizer = CountVectorizer(ngram_range=(1, 3))
      self.tfidf_transformer = TfidfTransformer()
      counts = self.count_vectorizer.fit_transform(df["text"])
      features = self.tfidf_transformer.fit_transform(counts)
      self.first_run = False
    else:
      counts = self.count_vectorizer.transform(df["text"])
      features = self.tfidf_transformer.transform(counts)

    return features