import numpy as np
from feature.features import Features
import re

class LengthFeatures(Features):
  def __init__(self):
    super().__init__('napoles/length_features')

  def _extract_features(self, df):
    features = [
        df['text'].apply(lambda x: len(re.split(r'[.!?]+', x))),
        df['text'].apply(lambda x: len(x.split(' '))),
    ]

    return np.vstack(features).T