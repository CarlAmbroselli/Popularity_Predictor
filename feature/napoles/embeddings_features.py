import numpy as np
from gensim.models import Word2Vec as gensim_word2vec
from feature.features import Features
import gensim
import re

class EmbeddingsFeatures(Features):
  def __init__(self):
    super().__init__('napoles/embeddings_features')
    self.w2v_model = None

  def initialize_variables(self):
    # load model
    self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format('model/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

  def _extract_features(self, df):
    if self.w2v_model is None:
      self.initialize_variables()
    data = df['text'].apply(self.text_to_wordlist)
    vectors = data.apply(self.acticle_to_vectors)
    return np.vstack(vectors)

  def text_to_wordlist(self, acticle):
    try:
      acticle_text = re.sub("[^a-zA-ZöÖüÜäÄß]", " ", acticle)
      acticle_text = re.sub("\s\s+", " ", acticle_text)
    except:
      acticle_text = ''
    return acticle_text

  def word_to_position(self, word):
    try:
      return self.w2v_model.wv[word]
    except:
      return -1

  def acticle_to_vectors(self, acticle):
    words = acticle.split(' ')
    result = list(map(self.word_to_position, words))
    result = sum(result) / len(words)
    return result

