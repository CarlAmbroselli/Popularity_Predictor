import numpy as np
from feature.features import Features
import spacy

class EntityFeatures(Features):
  def __init__(self):
    super().__init__('napoles/entity_features')
    self.nlp = None

  def _extract_features(self, df):
    if not self.nlp:
      self.nlp = spacy.load('de')

    features = df['text'].apply(lambda x: self.count_entities(str(x)))
    features = np.vstack(features)

    return features

  def count_entities(self, text):
    counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    doc = self.nlp(text)
    for ent in doc.ents:
      if ent.label_ == 'PERSON': # People, including fictional.
        counts[0] += 1
      if ent.label_ == 'NORP': # Nationalities or religious or political groups.
        counts[1] += 1
      if ent.label_ == 'FACILITY': # Buildings, airports, highways, bridges, etc.
        counts[2] += 1
      if ent.label_ == 'ORG': # Companies, agencies, institutions, etc.
        counts[3] += 1
      if ent.label_ == 'GPE': # Countries, cities, states.
        counts[4] += 1
      if ent.label_ == 'LOC': # Non-GPE locations, mountain ranges, bodies of water.
        counts[5] += 1
      if ent.label_ == 'PRODUCT': # Objects, vehicles, foods, etc. (Not services.)
        counts[6] += 1
      if ent.label_ == 'EVENT': # Named hurricanes, battles, wars, sports events, etc.
        counts[7] += 1
      if ent.label_ == 'WORK_OF_ART': # Titles of books, songs, etc.
        counts[8] += 1
      if ent.label_ == 'LANGUAGE': # Any named language.
        counts[9] += 1
      if ent.label_ == 'DATE': # Absolute or relative dates or periods.
        counts[10] += 1
      if ent.label_ == 'TIME': # Times smaller than a day.
        counts[11] += 1
      if ent.label_ == 'PERCENT': # Percentage, including "%".
        counts[12] += 1
      if ent.label_ == 'MONEY': # Monetary values, including unit.
        counts[13] += 1
      if ent.label_ == 'QUANTITY': # Measurements, as of weight or distance.
        counts[14] += 1
      if ent.label_ == 'ORDINAL': # "first", "second", etc.
        counts[15] += 1
      if ent.label_ == 'CARDINAL': # Numerals that do not fall under another type.
        counts[16] += 1
    return counts