import numpy as np
from feature.features import Features

class Features(Features):
  def __init__(self):
    super().__init__('carl/features')

  def _extract_features(self, df):
    features = []

    year_month = (df['year'] - 2014)*12 + df['month']
    features.append(year_month)
    features.append(year_month ** 2)    

    year_month_day = (df['year'] - 2014)*12*31 + df['month']*12 + df['day']
    features.append(year_month_day)
    features.append(year_month_day ** 2)

    publish_datestring = df['publish_datestring']
    features.append(publish_datestring)

    return np.vstack(features).T
