import numpy as np
import spacy
from feature.features import Features
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack as sparse_hstack
import operator
import code

class SemanticFeatures(Features):
  def __init__(self):
    super().__init__('bandari/semantic_features')
    self.nlp = None
    self.ne_vocabulary = None
    self.topwords = None

  def _extract_features(self, df):
    if not self.nlp:
      self.nlp = spacy.load('de')

    teaser_features = np.vstack(df['teaser_text'].apply(lambda x: self.count_entities(str(x))))
    title_features = np.vstack(df['title'].apply(lambda x: self.count_entities(str(x))))

    used_by_paper = [np.vstack(np.sum(teaser_features, axis=1)), np.vstack(np.max(teaser_features, axis=1)), np.vstack(np.average(teaser_features, axis=1)), np.vstack(np.sum(title_features, axis=1)), np.vstack(np.max(title_features, axis=1)), np.vstack(np.average(title_features, axis=1))]
    return np.hstack(used_by_paper)


  def named_entities_tfidf(self, articles):
    count_vect = CountVectorizer(vocabulary=self.named_entities_list())
    tfidf_transformer = TfidfTransformer()
    counts = count_vect.fit_transform(articles)
    tfidf = tfidf_transformer.fit_transform(counts)
    return tfidf

  def top_entities(self):
    if self.topwords is not None:
      return self.topwords
    filepath = 'feature/cache/top_named_entities_per_category.pickle'
    if os.path.isfile(filepath):
      return pickle.load(open(filepath, 'rb'))
    else:
      nlp = spacy.load('de')
      vocabulary = {}

      articles = pd.read_csv('data/datasets/Tr09-16Te17/train/articles.csv', sep=',')['text']
      print('read articles')

      for doc in nlp.pipe(articles, batch_size=1000, n_threads=25):
        for ent in doc.ents:
          if ent.label_ not in vocabulary:
            vocabulary[ent.label_] = {}
          if ent.text.lower() not in vocabulary[ent.label_]:
            vocabulary[ent.label_][ent.text.lower()] = 1
          else:
            vocabulary[ent.label_][ent.text.lower()] += 1

      topwords = {}
      print('extracting topwords...')
      for key, value in vocabulary.items():
        topwords[key] = list(dict(sorted(value.items(), key=operator.itemgetter(1), reverse=True)[:50]).keys())

      pickle.dump(topwords, open(filepath, 'wb'))
      self.topwords = topwords
      return self.topwords

  def count_entities(self, text):
    counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    doc = self.nlp(text)
    top_entities = self.top_entities()
    top_entities_count = [0] * 150
    for ent in doc.ents:
      if ent.text in top_entities['LOC']:
        top_entities_count[top_entities['LOC'].index(ent.text)] += 1
      if ent.text in top_entities['ORG']:
        top_entities_count[top_entities['ORG'].index(ent.text) + 50] += 1
      if ent.text in top_entities['PERSON']:
        top_entities_count[top_entities['PERSON'].index(ent.text) + 100] += 1
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
    return counts + top_entities_count

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

      articles = pd.read_csv('data/datasets/Tr09-16Te17/train/articles.csv', sep=',')['text']
      for doc in nlp.pipe(articles, batch_size=1000, n_threads=25):
        for ent in doc.ents:
          vocabulary.add(ent.text.lower())
      voc = list(vocabulary)
      pickle.dump(voc, open(filepath, 'wb'))
      self.ne_vocabulary = voc
      return voc


