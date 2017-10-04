import numpy as np
import pandas as pd
from feature.features import Features
from textblob_de import TextBlobDE

class SubjectivityFeatures(Features):
  def __init__(self):
    super().__init__('subjectivity_features')

  def _extract_features(self, df):
    features = df['text'].apply(lambda x: [self.extract_subjectivity(str(x))])
    return np.vstack(features)

  def extract_subjectivity(self, text):
      blob = TextBlobDE(text)
      subjectivity = np.average([sentence.sentiment.subjectivity for sentence in blob.sentences])

      features = [sentence.sentiment.subjectivity for sentence in blob.sentences][:30]
      features += [subjectivity] * (30 - len(features))
      return np.append(features, [[subjectivity]])