import numpy as np
from feature.features import Features

class CumulativeFeatures(Features):
  def __init__(self):
    super().__init__('cumulative_features')

  def _extract_features(self, df):
    features = []
    counts = df.groupby(['publish_datestring']).count()['url']

    art_same_hr = df['publish_datestring'].apply(lambda x: counts[x])
    features.append(art_same_hr)

    # The following features were not implemented:
    # dupes_int_cnt, dupes_ext_cnt

    return np.vstack(features).T
