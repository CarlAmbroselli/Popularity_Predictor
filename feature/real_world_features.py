import numpy as np
from feature.features import Features

class RealWorldFeatures(Features):
  def __init__(self):
    super().__init__('real_world_features')

  def _extract_features(self, df):
    # weather from http://www.dwd.de/DE/leistungen/klimadatendeutschland/klimadatendeutschland.html

    features = [
        df['temp_ham'],
        df['temp_fra'],
        df['temp_ber'],
        df['hum_ham'],
        df['hum_fra'],
        df['hum_ber'],
    ]

    return np.vstack(features).T
