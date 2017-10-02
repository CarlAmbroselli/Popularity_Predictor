import numpy as np
import spacy
from feature.features import Features
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack as sparse_hstack

class SemanticFeatures(Features):
  def __init__(self):
    super().__init__('semantic_features')
    self.nlp = None
    self.ne_vocabulary = None

  def _extract_features(self, df):
    if not self.nlp:
      self.nlp = spacy.load('de')

    features = df['text'].apply(lambda x: self.count_entities(str(x)))
    tfidf_features = self.named_entities_tfidf(df['text'])

    return sparse_hstack((np.vstack(features), tfidf_features))

  def named_entities_tfidf(self, articles):
    count_vect = CountVectorizer(vocabulary=self.named_entities_list())
    tfidf_transformer = TfidfTransformer()
    counts = count_vect.fit_transform(articles)
    tfidf = tfidf_transformer.fit_transform(counts)
    return tfidf

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

  def named_entities_list(self):
    if self.ne_vocabulary:
      return self.ne_vocabulary
    filepath = 'feature/cache/named_entities_vocabulary.pickle'
    if os.path.isfile(filepath):
      self.ne_vocabulary = pickle.load(open(filepath, 'rb'))
      return self.ne_vocabulary
    else:
      nlp = spacy.load('de')
      vocabulary = set()

      articles = pd.read_csv('../data/datasets/all/articles.csv', sep=',')['text']
      for doc in nlp.pipe(articles, batch_size=1000, n_threads=25):
        for ent in doc.ents:
          vocabulary.add(ent.text.lower())
      voc = list(vocabulary)
      pickle.dump(voc, open(filepath, 'wb'))
      self.ne_vocabulary = voc
      return voc


