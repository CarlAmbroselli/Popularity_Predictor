import numpy as np
from feature.features import Features

class AfterPublicationFeatures(Features):
  def __init__(self, maximum_time):
    super().__init__('carl/after_publication_features?' + str(maximum_time))
    self.maximum_time = maximum_time

  def _extract_features(self, df):
    features = []
    if self.maximum_time == 2:
        features.append(df['comments_after_2'])
    if self.maximum_time == 4:
        features.append(df['comments_after_4'])
    if self.maximum_time == 8:
        features.append(df['comments_after_8'])
    if self.maximum_time == 16:
        features.append(df['comments_after_16'])
    if self.maximum_time == 32:
        features.append(df['comments_after_32'])
    if self.maximum_time == 64:
        features.append(df['comments_after_64'])
    if self.maximum_time == 128:
        features.append(df['comments_after_128'])
    if self.maximum_time == 256:
        features.append(df['comments_after_256'])
    if self.maximum_time == 512:
        features.append(df['comments_after_512'])
    if self.maximum_time == 1024:
        features.append(df['comments_after_1024'])
    if self.maximum_time == 2048:
        features.append(df['comments_after_2048'])
    if self.maximum_time == 4096:
        features.append(df['comments_after_4096'])
    if self.maximum_time == 8192:
        features.append(df['comments_after_8192'])
    if self.maximum_time == 16384:
        features.append(df['comments_after_16384'])
    if self.maximum_time == 32768:
        features.append(df['comments_after_32768'])
    if self.maximum_time == 65536:
        features.append(df['comments_after_65536'])
    if self.maximum_time == 131072:
        features.append(df['comments_after_131072'])
    if self.maximum_time == 262144:
        features.append(df['comments_after_262144'])
    if self.maximum_time == 524288:
        features.append(df['comments_after_524288'])
    if self.maximum_time == 1048576:
        features.append(df['comments_after_1048576'])

    return np.vstack(features).T
