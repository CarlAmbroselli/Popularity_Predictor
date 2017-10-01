import spacy
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

filepath = '../feature/cache/named_entities_vocabulary.pickle'
if os.path.isfile(filepath):
  # return pickle.load(open(filepath, 'rb'))
  pass
else:
  nlp = spacy.load('de')
  count_vect = CountVectorizer()
  tfidf_transformer = TfidfTransformer()
  vocabulary = set()

  articles = pd.read_csv('../data/datasets/all/articles.csv', sep=',')['text']
  for doc in nlp.pipe(articles, batch_size=1000, n_threads=25):
    for ent in doc.ents:
      vocabulary.add(vocabulary.text.lower())
  voc = list(vocabulary)
  pickle.dump(voc, open(filepath,'wb'))
    
