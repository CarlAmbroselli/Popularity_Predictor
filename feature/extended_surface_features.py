import numpy as np
from feature.features import Features

class ExtendedSurfaceFeatures(Features):
  def __init__(self):
    super().__init__('extended_surface_features')

  def _extract_features(self, df):
    features = []

    hour = df['hour']
    features.append(hour)

    month = df['month']

    day = df['day']
    features.append(month * 30 + day)

    year = df['year']
    features.append(year)

    publish_datestring = df['publish_datestring']
    features.append(publish_datestring)

    return np.vstack(features).T
