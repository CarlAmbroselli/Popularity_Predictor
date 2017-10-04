import numpy as np
from feature.features import Features
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import vstack as sparse_vstack
from scipy.sparse import csr_matrix
import pandas as pd
import os
import pickle

class MetaFeatures(Features):
  def __init__(self):
    super().__init__('meta_features')
    self.labels = None
    self._articles = None

  def _extract_features(self, df):

    encoder = self.label_encoder()

    features = [
        sparse_vstack(df['author'].apply(lambda x: self.hot_encoding(str(x).split(', '), encoder['author'])).as_matrix()),
        sparse_vstack(df['source'].apply(lambda x: self.hot_encoding([str(x)], encoder['source'])).as_matrix()),
        sparse_vstack(df['tags'].apply(lambda x: self.hot_encoding(str(x).split(', '), encoder['tags'])).as_matrix()),
        np.vstack(df['video_count'].as_matrix()),
        np.vstack(df['gallery_text'].apply(lambda x: str(x) != 'nan').as_matrix())
    ]

    return sparse_hstack(features)

  def label_encoder(self):
      if self.labels is None:
          filepath = 'feature/cache/meta_labels.pickle'
          if os.path.isfile(filepath):
              self.labels = pickle.load(open(filepath, 'rb'))
          else:
              articles = pd.read_csv('data/datasets/Tr09-16Te17/train/articles.csv', sep=',')
              items = np.array([
                  np.unique(np.concatenate(articles['author'].apply(lambda x: str(x).split(', ')))),
                  np.unique(articles['source'].apply(lambda x: str(x).strip()).as_matrix()),
                  np.unique(np.concatenate(articles['tags'].apply(lambda x: str(x).split(' , '))))
              ])
              self.labels = {
                  'author': {k: v for v, k in enumerate(items[0])},
                  'source': {k: v for v, k in enumerate(items[1])},
                  'tags': {k: v for v, k in enumerate(items[2])}
              }
              pickle.dump(self.labels, open(filepath, 'wb'))

      return self.labels

  def hot_encoding(self, values, labels):
      output = [0] * len(labels)
      for v in values:
          try:
            output[labels[v]] = 1
          except KeyError:
            pass
      return csr_matrix([output])

