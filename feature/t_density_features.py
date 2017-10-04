import numpy as np
import pandas as pd
from feature.features import Features

class TDensityFeatures(Features):
  def __init__(self):
    super().__init__('t_density_features')
    self.t_density = None

  def _extract_features(self, df):
    t_density_score = self.t_density_score(df)
    features = df['ressort'].apply(lambda x: \
        [(t_density_score[str(x)] if str(x) in t_density_score else t_density_score['avg']), t_density_score['avg']])

    return  np.vstack(np.array(features).T)

  def t_density_score(self, df):
      if self.t_density is not None:
          return self.t_density
      self._articles = df
      self.t_density = self._articles.groupby('ressort')['url'].count()
      for key in self.t_density.keys():
          self.t_density[key] = np.sum(self._articles[self._articles['ressort'] == key]['facebook_shares']) / self.t_density[key]

      self.t_density['nan'] = np.average(self._articles['facebook_shares'])
      self.t_density['avg'] = self.t_density['nan']

      return self.t_density

  def reset(self):
      super().reset()
      self._articles = None
      self.t_density = None