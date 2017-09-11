import numpy as np

class RealWorldFeatures:

  def extract_features(self, df):

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
