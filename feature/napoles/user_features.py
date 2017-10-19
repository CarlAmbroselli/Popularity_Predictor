import numpy as np
from feature.features import Features
import pandas as pd

class UserFeatures(Features):
  def __init__(self):
    super().__init__('napoles/user_features')
    self.users_df = None

  def _extract_features(self, df):
    if self.users_df is None:
      self.users_df = pd.read_csv('data/datasets/YNACC/users.csv', sep=',', index_col='index')
    features = df['guid'].apply(lambda x: self.users_df.loc[x])

    return features.values
