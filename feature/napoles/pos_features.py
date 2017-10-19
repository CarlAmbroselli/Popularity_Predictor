import numpy as np
from feature.features import Features
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import spacy

class POSFeatures(Features):
  def __init__(self):
    super().__init__('napoles/pos_features')
    self.nlp = None
    self.first_run = True

  def _extract_features(self, df):
    if self.first_run:
      self.nlp = spacy.load('en')
    pos = df['text'].apply(lambda x: ' '.join([word.pos_ for word in self.nlp(x)]))

    if self.first_run:
      self.count_vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2)
      self.tfidf_transformer = TfidfTransformer()
      counts = self.count_vectorizer.fit_transform(pos)
      features = self.tfidf_transformer.fit_transform(counts)
      self.first_run = False
    else:
      counts = self.count_vectorizer.transform(pos)
      features = self.tfidf_transformer.transform(counts)
    return features

  def reset(self):
    self.first_run = True