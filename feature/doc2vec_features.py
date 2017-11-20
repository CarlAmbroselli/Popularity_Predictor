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

class Doc2VecFeatures(Features):
  def __init__(self, num_dimensions):
    super().__init__('doc2vec_features?' + str(num_dimensions))
    self.model = None
    self.num_dimensions = num_dimensions

  def _extract_features(self, df):
    model = self.doc2vec_model()
    documents = self.get_doc(df['text'], return_tagged_documents=False)

    features = np.array([model.infer_vector(doc_words=doc, steps=20) for doc in documents])

    return features

  def doc2vec_model(self):
    if self.model is not None:
      return self.model
    filepath = 'feature/cache/doc2vec_' + str(self.num_dimensions)
    if os.path.isfile(filepath):
      self.model = doc2vec.Doc2Vec.load(filepath)
      return self.model
    else:
      print('Recalculating Doc2Vec Model')
      print('Loading documents')
      doc_list = pd.read_csv('data/datasets/all/articles.csv', sep=',')['text']
      documents = self.get_doc(doc_list)
      cores = 20 if multiprocessing.cpu_count() > 4 else 3
      print('Training model using {} cores'.format(cores))
      model = doc2vec.Doc2Vec(documents, size=self.num_dimensions, window=8, min_count=5, workers=cores, iter=20)
      # model.save(filepath)
      model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
      model.save(filepath)
      self.model = model
      return self.model

  def get_doc(self, doc_list, return_tagged_documents=True, show_progress=False):
    tokenizer = RegexpTokenizer(r'\w+')
    de_stop = stopwords.words('german')
    stemmer = SnowballStemmer("german")

    taggeddoc = []

    texts = []
    if show_progress:
      bar = progressbar.ProgressBar(max_value=len(doc_list))
    for index, i in enumerate(doc_list):
      if show_progress:
        bar.update(index)
      # clean and tokenize document string
      raw = i.lower()
      tokens = tokenizer.tokenize(raw)

      # remove stop words from tokens
      stopped_tokens = [i for i in tokens if not i in de_stop]

      # remove numbers
      number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
      number_tokens = ' '.join(number_tokens).split()

      # stem tokens
      stemmed_tokens = [stemmer.stem(i) for i in number_tokens]
      # remove empty
      length_tokens = [i for i in stemmed_tokens if len(i) > 1]
      # add tokens to list
      texts.append(length_tokens)

      if return_tagged_documents:
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(), str(index))
      else:
        td = gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split()
      taggeddoc.append(td)

    return taggeddoc


