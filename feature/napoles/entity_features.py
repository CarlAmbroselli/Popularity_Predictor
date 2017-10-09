import numpy as np
from feature.features import Features

class EntityFeatures(Features):
  def __init__(self):
    super().__init__('napoles/entity_features')

  def _extract_features(self, df):
    features = [
        np.ones(df.shape[0])
    ]

    return np.vstack(features).T
