from gensim.models import Word2Vec as gensim_word2vec
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import re
import pandas as pd
from feature.features import Features
from feature.helper import Helper
import os
 
class Word2Vec(Features):
    def __init__(self, num_dimensions=100):
        super().__init__('word2vec?' + str(num_dimensions))
        self.first_run = True
        self.num_dimensions = num_dimensions

    def initialize_variables(self, df):
        # load model
        language = 'de'
        filepath = 'model/word2vec/all_lowercased_stemmed_' + str(self.num_dimensions) + '_' + language
        if os.path.isfile(filepath):
            self.w2v_model = gensim_word2vec.load(filepath)
        else:
            doc_list = pd.read_csv('data/datasets/all/articles.csv', sep=',')['text']
            print('Preprocessing new word2vec model')
            texts = Helper.remove_stop_and_stem(df['text']).apply(lambda x: str(x).split(' '))
            print('Training new word2vec model')
            self.w2v_model = gensim_word2vec(texts, size=self.num_dimensions, window=5, min_count=5, workers=20)
            self.w2v_model.save(filepath)

    def _extract_features(self, df):
        if self.first_run:
            self.initialize_variables(df)
            self.first_run = False
        data = [Helper.remove_stop_and_stem(x) for x in [
            df['next_read_title'],
            df['next_read_kicker'],
            df['title'],
            df['supertitle'],
            df['teaser_text'],
            df['teaser_title']
        ]]

        vectors = [np.vstack(d.apply(self.acticle_to_vectors)) for d in data]
        return np.hstack(vectors)

    def text_to_wordlist(self, acticle):
        try:
            acticle_text = re.sub("[^a-zA-ZöÖüÜäÄß]"," ", acticle)
            acticle_text = re.sub("\s\s+"," ", acticle_text)
            # acticle_text = re.sub(r'https?:\/\/.*[\r\n]*', '', acticle, flags=re.MULTILINE)
            # acticle_text = re.sub(r'<\/?em>', '', acticle_text, flags=re.MULTILINE)
        except:
            acticle_text = ''
        return acticle_text

    def word_to_position(self, word):
        try:
            return self.w2v_model.wv[word]
        except:
            return np.array([-1] * self.num_dimensions)

    def acticle_to_vectors(self, acticle):
        words = acticle.split(' ')
        result = list(map(self.word_to_position, words))
        result = sum(result) / len(words)
        return result

