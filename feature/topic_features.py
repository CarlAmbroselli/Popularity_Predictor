import numpy as np
from feature.features import Features
import os
from nltk.corpus import stopwords
import pandas as pd
import multiprocessing
import gensim
import os
import re
from gensim.models import doc2vec
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim.models.doc2vec import TaggedDocument
import progressbar

class TopicFeatures(Features):
  def __init__(self):
    super().__init__('topic_features')
    self.lda = None
    self.dict = None

  def _extract_features(self, df):
    dict, lda = self.topic_model()
    data = df['text'].apply(lambda x: Helper.remove_stop_and_stem(x))
    bow_data = data.apply(lambda x: dict.doc2bow(x))
    features = bow_data.apply(lambda x: lda.get_document_topics(x, minimum_probability=0))

    return features

  def topic_model(self):
    if self.model is not None:
      return self.model
    topic_filepath = 'feature/cache/topicmodel'
    dict_filepath = 'feature/cache/topicmodel_dict'
    if os.path.isfile(topic_filepath) and os.path.isfile(dict_filepath):
      self.dict = corpora.Dictionary().load('model/ldamodel/dictionary.dict')
      self.lda = models.LdaModel.load('model/ldamodel/lda.model')
      return (self.dict, self.lda)
    else:
      print('Recalculating model')
      doc_list = pd.read_csv('data/datasets/Tr09-16Te17/train/articles.csv', sep=',')['text']
      articles = Helper.remove_stop_and_stem(doc_list)
      dictionary = corpora.Dictionary(articles)
      dictionary.save(dict_filepath)
      corpus = [dictionary.doc2bow(text) for text in articles]
      self.dict = dictionary

      print('Starting training')
      self.lda = models.ldamodel.LdaModel(corpus, num_topics=200, alpha='auto')
      # save the trained model
      self.lda.save(dict_filepath)
      print('Finished training')
      return(self.dict, self.lda)

