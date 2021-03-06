import numpy as np
from feature.features import Features
import os.path
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import pandas as pd

class TextFeatures(Features):
  def __init__(self):
    super().__init__('text_features')

  def _extract_features(self, df):

    text = df['text'].apply(lambda x: str(x))
    total_length = text.apply(lambda x: len(x))
    num_of_words = text.apply(lambda x: len(x.split()))
    avg_length = text.apply(lambda x: np.average([len(a) for a in x.split()]))
    num_questions = text.apply(lambda x: x.count('?'))
    num_quote = text.apply(lambda x: x.count('"'))
    num_dot = text.apply(lambda x: x.count('.'))
    num_repeated_dot = text.apply(lambda x: x.count('..'))
    num_exclamation = text.apply(lambda x: x.count('!'))
    num_http = text.apply(lambda x: x.count('http'))
    num_negations = text.apply(lambda x: x.count('nicht') + x.count('nie') + x.count('weder') + x.count('nichts'))
    ratio_capitalized = text.apply(lambda x: sum(1 for c in x if c.isupper()) / len(x))

    text_features = np.vstack((
      total_length,
      num_of_words,
      avg_length,
      num_questions,
      num_quote,
      num_dot,
      num_repeated_dot,
      num_exclamation,
      num_http,
      num_negations,
      ratio_capitalized
    )).T

    return text_features