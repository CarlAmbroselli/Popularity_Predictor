from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from feature.features import Features
import os
import pickle
from feature.helper import Helper

class NGramFeatures(Features):
  def __init__(self, ngram_range=(1,3), stem=False):
    super().__init__('ngram_features?' + str(ngram_range) + '*' + str(stem))
    self.first_run = True
    self.ngram_range=ngram_range
    self.stem=stem

  def load_cached_object(self, filepath):
    path = 'feature/cache/' + filepath
    if os.path.isfile(path):
      try:
        return pickle.load(open(path, 'rb'))
      except:
        return None
    else:
      return None

  def cache(self, object, filepath):
    path = 'feature/cache/' + filepath
    pickle.dump(object, open(path, 'wb'))

  def _extract_features(self, df):

    self.count_vectorizer = self.load_cached_object('stemmed_' + str(self.stem) + '_ngram_count_vectorizer_min_2_' + str(self.ngram_range))
    self.tfidf_transformer = self.load_cached_object('stemmed_' + str(self.stem) + '_ngram_tfidf_vectorizer_min_2_' + str(self.ngram_range))

    if self.stem:
      input = Helper.remove_stop_and_stem(df["text"])
    else:
      input = df["text"]
    if self.count_vectorizer == None:
      self.count_vectorizer = CountVectorizer(ngram_range=self.ngram_range, min_df=2, stop_words=stopwords.words('german'))
      self.tfidf_transformer = TfidfTransformer()
      self.count_vectorizer.fit(input)
      self.cache(self.count_vectorizer, 'stemmed_' + str(self.stem) + '_ngram_count_vectorizer_min_2_' + str(self.ngram_range))
      counts = self.count_vectorizer.transform(input)
      # pickle.dump(self.count_vectorizer.vocabulary_, open( "ngram_vocabulary_" + str(self.ngram_range) + ".pickle", "wb" ), protocol=4)
      self.tfidf_transformer.fit(counts)
      self.cache(self.tfidf_transformer, 'stemmed_' + str(self.stem) + '_ngram_tfidf_vectorizer_min_2_' + str(self.ngram_range))
      features = self.tfidf_transformer.transform(counts)
      self.first_run = False
    else:
      print("Using cached vectorizers")
      counts = self.count_vectorizer.transform(input)
      features = self.tfidf_transformer.transform(counts)
    return features

  def reset(self):
    self.first_run = True
