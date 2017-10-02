import numpy as np
import pandas as pd
from feature.features import Features
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import itertools

class CumulativeFeatures(Features):
  def __init__(self):
    super().__init__('cumulative_features')
    self.article_tfidf = None
    self.cached_articles = None

  def _extract_features(self, df):
    features = []
    counts = df.groupby(['publish_datestring']).count()['url']

    art_same_hr = df['publish_datestring'].apply(lambda x: counts[x])
    features.append(art_same_hr)

    dupes_int_cnt = df['url'].apply(lambda x: self.near_duplicates(x))

    # The following features were not implemented:
    # dupes_ext_cnt

    return np.hstack((np.vstack(features).T, np.vstack(dupes_int_cnt)))

  def near_duplicates(self, url):
    articles =  self.articles()
    index = articles['url'].searchsorted(url)
    similarity_matrix = self.pairwise_similarity()
    similarity = similarity_matrix * similarity_matrix[index].T
    return [np.sum(similarity > 0.4), np.sum(similarity > 0.5), np.sum(similarity > 0.6)]

  def articles(self):
    if self.cached_articles is None:
      self.cached_articles = pd.read_csv('data/datasets/all/articles.csv', sep=',')
      self.cached_articles = self.cached_articles.sort_values('url')
    return self.cached_articles

  def pairwise_similarity(self):
    if self.article_tfidf is not None:
      return self.article_tfidf

    filepath = 'feature/cache/article_tfidf.pickle'
    if os.path.isfile(filepath):
      self.article_tfidf = pickle.load(open(filepath, 'rb'))
      return self.article_tfidf
    else:
      articles = self.articles()
      print("Calculate tfidf")
      tfidf = TfidfVectorizer().fit_transform(articles['text'])
      pickle.dump(tfidf, open(filepath, 'wb'))
      # no need to normalize, since Vectorizer will return normalized tf-idf
      self.article_tfidf = tfidf
      return self.article_tfidf