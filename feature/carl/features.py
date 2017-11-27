import numpy as np
from feature.features import Features
import ciso8601

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

    weekday = df['date_first_released'].apply(lambda x: ciso8601.parse_datetime(x).weekday())

    for i in range(0,7):
      features.append(weekday == i)

    for i in range(1,13):
      features.append(df['month'] == i)

    for i in range(1,25):
      features.append(df['hour'] == i)

    features.append(df['year'])

    # publish_datestring = df['publish_datestring']
    # features.append(publish_datestring)

    return np.vstack(features).T
