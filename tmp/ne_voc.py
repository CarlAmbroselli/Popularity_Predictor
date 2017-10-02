import spacy
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import operator

filepath = '../feature/cache/top_named_entities_per_category.pickle'
if os.path.isfile(filepath):
  # return pickle.load(open(filepath, 'rb'))
  pass
else:
  nlp = spacy.load('de')
  count_vect = CountVectorizer()
  tfidf_transformer = TfidfTransformer()
  vocabulary = {}

  articles = pd.read_csv('../data/datasets/all/articles.csv', sep=',')['text']
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

  pickle.dump(topwords, open(filepath,'wb'))
  print(topwords)
