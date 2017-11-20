from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from feature.features import Features



class NGramFeatures(Features):
  def __init__(self, ngram_range=(1, 3)):
    super().__init__('ngram_features?' + str(ngram_range))
    self.ngram_range = ngram_range
    self.first_run = True

  def _extract_features(self, df):
    if self.first_run:
      self.count_vectorizer = CountVectorizer(ngram_range=self.ngram_range, min_df=2, stop_words=stopwords.words('german'))
      self.tfidf_transformer = TfidfTransformer()
      counts = self.count_vectorizer.fit_transform(df["text"])
      features = self.tfidf_transformer.fit_transform(counts)
      self.first_run = False
    else:
      counts = self.count_vectorizer.transform(df["text"])
      features = self.tfidf_transformer.transform(counts)
    return features

  def reset(self):
    self.first_run = True
