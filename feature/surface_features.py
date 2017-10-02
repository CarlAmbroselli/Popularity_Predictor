import numpy as np
from feature.features import Features

class SurfaceFeatures(Features):
  def __init__(self):
    super().__init__('surface_features')

  def _extract_features(self, df):
    features = []
    text = df['text'].apply(lambda x: str(x))

    month = df['month']
    features.append(month)

    wom = (df['release_date'].astype("datetime64[ns]").dt.day-1)//7+1
    features.append(wom)

    dow = df['release_date'].astype("datetime64[ns]").dt.dayofweek
    features.append(dow)

    day = df['day']
    features.append(day)

    hour = df['hour']
    features.append(hour)

    first_half_hour = df['minute'] < 30
    features.append(first_half_hour)

    art_char_length = text.apply(lambda x: len(x))
    features.append(art_char_length)

    num_categories = df['tags'].apply(lambda x: str(x).count(',')) + 1
    features.append(num_categories)

    authors_cnt = df['author'].apply(lambda x: (len(str(x)) > 0) + str(x).count(',')) + 1
    features.append(authors_cnt)

    # The following features were not implemented:
    # has_summary, has_content, has_content_clean, links_cnt

    return np.vstack(features).T
