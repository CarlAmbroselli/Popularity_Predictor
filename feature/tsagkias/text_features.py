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
    super().__init__('tsagkias/text_features')

  def _extract_features(self, df):
    top_term_tf = self.load_tf()

    df = df.copy(deep=True)
    df['ressort_scraped'].fillna('unknown', inplace=True)

    features = df.apply(lambda x: self.extract_term_frequencies(str(x['text']), top_term_tf[x['ressort_scraped']]), axis=1)

    return np.array([x[0] for x in features])

  def extract_term_frequencies(self, text, term_tfs):
    term_dict = dict(zip(term_tfs, [0]*len(term_tfs)))
    words = nltk.word_tokenize(text)
    for word in words:
      if word in term_dict:
        term_dict[word] += 1

    frequencies = [np.array([v for (k,v) in term_dict.items()])]
    return frequencies

  def load_tf(self):
    filepath = 'feature/cache/top_tf.pickle'
    if os.path.isfile(filepath):
      return pickle.load(open(filepath, 'rb'))
    else:
      count_vect = CountVectorizer()
      tfidf_transformer = TfidfTransformer()

      articles = pd.read_csv('data/datasets/Tr14-16Te17/train/articles.csv', sep=',')[['text', 'ressort']]
      articles['ressort'].fillna('unknown', inplace=True)

      # Extract urlressort
      # ==========================
      # def extract_url(url):
      #   search =  re.search("it.de/([^/]+)/.*", url)
      #   if search:
      #     return search.group(1)
      #   else:
      #     return 'unknown'

      # articles['urlressort'] = articles['url'].apply(lambda x: extract_url(x))
      # ==========================

      # sub_articles = articles[articles['publish_datestring'] > 2016010100]

      counts = count_vect.fit_transform(articles['text'])
      vocabulary = count_vect.vocabulary_
      tfidf = tfidf_transformer.fit_transform(counts)
      means = tfidf.mean(axis=0)

      top_tf = {}

      for label, group in articles.groupby('ressort_scraped'):
        group_count_vect = CountVectorizer(vocabulary=vocabulary)
        group_counts = group_count_vect.fit_transform(group['text'])
        group_tfidf_transformer = TfidfTransformer()
        group_tfidf = group_tfidf_transformer.fit_transform(group_counts)
        group_means = group_tfidf.mean(axis=0)
        substraction = np.subtract(group_means[0], means[0])
        indices = np.argsort(substraction)
        index = np.asarray(indices)[0]
        strings = count_vect.get_feature_names()
        frequent_words = [strings[i] for i in index]
        print(label, '=>', frequent_words[:30])
        top_tf[label] = frequent_words[:100]

      pickle.dump(top_tf, open(filepath,'wb'))
      return top_tf
