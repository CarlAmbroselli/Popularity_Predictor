from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import re

class Helper():
    @staticmethod
    def text_to_wordlist(acticle):
        try:
            acticle_text = re.sub("[^a-zA-ZöÖüÜäÄß]"," ", acticle)
            acticle_text = re.sub("\s\s+"," ", acticle_text)
            # acticle_text = re.sub(r'https?:\/\/.*[\r\n]*', '', acticle, flags=re.MULTILINE)
            # acticle_text = re.sub(r'<\/?em>', '', acticle_text, flags=re.MULTILINE)
        except:
            acticle_text = ''
        return acticle_text

    @staticmethod
    def to_wordlist(data):
        return data.apply(text_to_wordlist)

    @staticmethod
    def remove_stopwords(data):
        stop = stopwords.words('german')
        return data.apply(lambda x: [item for item in str(x).split(' ') if item not in stop])

    @staticmethod
    def stem(data):
        return data.apply(lambda x: " ".join([stemmer.stem(y) for y in x]))

    @staticmethod
    def remove_stop_and_stem(data):
        data = to_wordlist(data)
        data = remove_stopwords(data)
        data = stem(data)
        return data