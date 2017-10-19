import numpy as np
from gensim import corpora

from feature.features import Features
import os
from nltk.corpus import stopwords
import pandas as pd
import multiprocessing
import gensim
import re
from gensim.models import doc2vec
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim.models.doc2vec import TaggedDocument
import progressbar
import gensim.models as models
from feature.helper import Helper

class TopicFeatures(Features):
  def __init__(self):
    super().__init__('topic_features')
    self.lda = None
    self.dict = None

  def _extract_features(self, df):
    dict, lda = self.topic_model()
    data = Helper.remove_stop_and_stem(df['text'])
    bow_data = data.apply(lambda x: dict.doc2bow(x.split(' ')))
    features = bow_data.apply(lambda x: lda.get_document_topics(x, minimum_probability=0))

    return np.array([[topic[1] for topic in f] for f in features]) # extract only the probabilities

  def topic_model(self):
    if self.dict is not None and self.lda is not None:
      return (self.dict, self.lda)
    topic_filepath = 'feature/cache/topicmodel'
    dict_filepath = 'feature/cache/topicmodel_dict'
    if os.path.isfile(topic_filepath) and os.path.isfile(dict_filepath):
      self.dict = corpora.Dictionary().load(dict_filepath)
      self.lda = models.LdaModel.load(topic_filepath)
      return (self.dict, self.lda)
    else:
      print('Recalculating model')
      doc_list = pd.read_csv('data/datasets/Tr09-16Te17/train/articles.csv', sep=',')['text']
      articles = Helper.remove_stop_and_stem(doc_list)
      dictionary = corpora.Dictionary(articles.apply(lambda x: x.split(' ')))
      dictionary.save(dict_filepath)
      corpus = [dictionary.doc2bow(text.split(' ')) for text in articles]
      self.dict = dictionary

      print('Starting training')
      self.lda = models.ldamodel.LdaModel(corpus, num_topics=200, alpha='auto')
      # save the trained model
      self.lda.save(topic_filepath)
      print('Finished training')
      return(self.dict, self.lda)