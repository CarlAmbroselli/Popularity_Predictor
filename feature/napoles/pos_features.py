import numpy as np
from feature.features import Features

class POSFeatures(Features):
  def __init__(self):
    super().__init__('napoles/pos_features')

  def _extract_features(self, df):
    features = [
        np.ones(df.shape[0])
    ]

    return np.vstack(features).T
