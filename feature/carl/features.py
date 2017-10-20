import numpy as np
from feature.features import Features

class Features(Features):
  def __init__(self):
    super().__init__('carl/features')

  def _extract_features(self, df):
    features = []

    year_month = (df['year'] - 2009)*12 + df['month']
    features.append(year_month)

    publish_datestring = df['publish_datestring']
    features.append(publish_datestring)

    return np.vstack(features).T
