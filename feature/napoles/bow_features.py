import numpy as np
from feature.features import Features
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class BowFeatures(Features):
  def __init__(self):
    self.first_run = True
    super().__init__('napoles/bow_features')

  def _extract_features(self, df):
    if self.first_run:
      self.count_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2)
      self.tfidf_transformer = TfidfTransformer()
      counts = self.count_vectorizer.fit_transform(df["text"])
      # features = self.tfidf_transformer.fit_transform(counts)
      self.first_run = False
    else:
      counts = self.count_vectorizer.transform(df["text"])
      # features = self.tfidf_transformer.transform(counts)
    return counts

  def reset(self):
    self.first_run = True
