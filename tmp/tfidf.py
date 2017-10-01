import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

articles = pd.read_csv('../data/datasets/all/articles.csv', sep=',')[['text', 'ressort']]

# Extract urlressort
# ==========================
# def extract_url(url):
#   search =  re.search("http://www.zeit.de/([^/]+)/.*", url)
#   if search:
#     return search.group(1)
#   else:
#     return 'unknown'

# articles['urlressort'] = articles['url'].apply(lambda x: extract_url(x))
# ==========================

sub_articles = articles[articles['publish_datestring'] > 2016010100]

counts = count_vect.fit_transform(sub_articles['text'])
vocabulary = count_vect.vocabulary_
tfidf = tfidf_transformer.fit_transform(counts)
means = tfidf.mean(axis=0)

top_tf_2016 = {}

for label, group in sub_articles.groupby('ressort'):
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
  top_tf_2016[label] = frequent_words[:100]

pickle.dump(top_tf_2016, open('../feature/cache/top_tf.pickle','wb'))