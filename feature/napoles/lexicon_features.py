import numpy as np
from feature.features import Features
import spacy

class LexiconFeatures(Features):
  def __init__(self):
    super().__init__('napoles/lexicon_features')
    self.nlp = None
    self.lexicon = None

  def _extract_features(self, df):
    if not self.nlp:
      self.nlp = spacy.load('en')

    features = df['text'].apply(lambda x: self.count_pronouns(str(x)))
    features = np.vstack(features)

    # not implemented: agreement and certainty phrases; discourse connectives; and abusive language.

    return features

  def count_pronouns(self, text):
    counts = 0
    doc = self.nlp(text)
    for word in doc:
      if word.pos_ == 'PRON':
        counts += 1
    return counts

  # def count_lexicon(self, text):
  #   pass
  #
  # def lexicon(self):
  #   if self.lexicon is None:
  #     self.lexicon = {
  #       'hallo': 2
  #     }
  #   return self.lexicon