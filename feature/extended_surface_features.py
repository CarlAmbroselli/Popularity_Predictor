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

    year_month = (df['year'] - 2009) * 12 + df['month']
    features.append(year_month)

    publish_datestring = df['publish_datestring']
    features.append(publish_datestring)

    return np.vstack(features).T
