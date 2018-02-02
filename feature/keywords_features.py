from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from feature.features import Features
import os
import pickle
from feature.helper import Helper

class KeywordFeatures(Features):
  def __init__(self, ngram_range=(1,1)):
    super().__init__('keywords_features?' + str(ngram_range))
    self.first_run = True
    self.ngram_range=ngram_range

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

    self.count_vectorizer = self.load_cached_object('keyword_count_vectorizer_min_2_' + str(self.ngram_range))
    self.tfidf_transformer = self.load_cached_object('keyword_tfidf_vectorizer_min_2_' + str(self.ngram_range))

    input =  df.tags.apply(lambda x: str(x)) #.apply(lambda x: ' '.join([y.replace(' ', '') for y in str(x).split(' , ')]))
    if self.count_vectorizer == None:
      self.count_vectorizer = CountVectorizer(ngram_range=self.ngram_range, min_df=2, stop_words=stopwords.words('german'))
      self.tfidf_transformer = TfidfTransformer()
      self.count_vectorizer.fit(input)
      self.cache(self.count_vectorizer, 'keyword_count_vectorizer_min_2_' + str(self.ngram_range))
      counts = self.count_vectorizer.transform(input)
      # pickle.dump(self.count_vectorizer.vocabulary_, open( "ngram_vocabulary_" + str(self.ngram_range) + ".pickle", "wb" ), protocol=4)
      self.tfidf_transformer.fit(counts)
      self.cache(self.tfidf_transformer, 'keyword_tfidf_vectorizer_min_2_' + str(self.ngram_range))
      features = self.tfidf_transformer.transform(counts)
      self.first_run = False
    else:
      print("Using cached vectorizers")
      counts = self.count_vectorizer.transform(input)
      features = self.tfidf_transformer.transform(counts)
    return features

  def reset(self):
    self.first_run = True
