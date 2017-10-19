import numpy as np
from feature.features import Features

class PopularityFeatures(Features):
  def __init__(self):
    super().__init__('napoles/popularity_features')

  def _extract_features(self, df):
    features = [
        df['thumbs-up'],
        df['thumbs-down'],
        df.apply(lambda x: x['thumbs-up'] + x['thumbs-down'], axis=1),
        df.apply(lambda x: x['thumbs-up'] / (x['thumbs-up'] + x['thumbs-down'] + 0.00001), axis=1), # to prevent division by zero
    ]

    return np.vstack(features).T
