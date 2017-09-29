from gensim.models import Word2Vec as gensim_word2vec
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import re
from feature.features import Features

class Word2Vec(Features):
  def __init__(self):
    super().__init__('word2vec')
    self.first_run = True

    def initialize_variables(self):
        # load model
        self.w2v_model = gensim_word2vec.load('model/word2vec/all_lowercased_stemmed')
        # initialize stemmer
        self.stemmer = SnowballStemmer('german')
        # grab stopword list
        self.stop = stopwords.words('german')

    def _extract_features(self, df):
        if self.first_run:
            self.initialize_variables()
            self.first_run = False
        data = self.remove_stop_and_stem(df['text'])
        vectors = np.asarray(list(map(self.acticle_to_vectors, data)))
        return vectors

    def text_to_wordlist(self, acticle):
        try:
            acticle_text = re.sub("[^a-zA-ZöÖüÜäÄß]"," ", acticle)
            acticle_text = re.sub("\s\s+"," ", acticle_text)
            # acticle_text = re.sub(r'https?:\/\/.*[\r\n]*', '', acticle, flags=re.MULTILINE)
            # acticle_text = re.sub(r'<\/?em>', '', acticle_text, flags=re.MULTILINE)
        except:
            acticle_text = ''
        return acticle_text

    def to_wordlist(self, data):
        return data.apply(self.text_to_wordlist)

    def remove_stopwords(self, data):
        return data.apply(lambda x: [item for item in str(x).split(' ') if item not in self.stop])

    def stem(self, data):
        return data.apply(lambda x: " ".join([self.stemmer.stem(y) for y in x]))

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

    def remove_stop_and_stem(self, data):
        data = self.to_wordlist(data)
        data = self.remove_stopwords(data)
        data = self.stem(data)
        return data

