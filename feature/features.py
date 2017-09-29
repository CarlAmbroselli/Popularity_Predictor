import hashlib
import numpy as np
import os.path
import pickle
from termcolor import colored


class Features:
    def __init__(self, file):
        self.file = file
        pass

    def extract_features(self, df):
        cached = self.load_cached(df)
        if cached is not None:
            return cached
        else:
            print('Recalculating:', colored(self.file, 'red', attrs=['bold']))
            features = self._extract_features(df)
            print(features)
            hash = self.filepath(df)
            pickle.dump(features, open(hash, 'wb'))
            return features

    def _extract_features(self, df):
        raise Exception('Unsupported Method', 'The child feature generator should overwrite this method.')

    def load_cached(self, df):
        filepath = self.filepath(df)
        if os.path.isfile(filepath):
            return pickle.load(open(filepath, 'rb'))
        else:
            return None

    def filepath(self, df):
        hash = hashlib.md5(''.join(str(x) for x in [df.shape, df.head(2), df.tail(2)]).encode('utf-8')).hexdigest()[:8]
        hasher = hashlib.md5()
        with open('feature/' + self.file +  '.py', 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        filehash = hasher.hexdigest()[:8]

        return 'feature/cache/' + hash + '_' + self.file + '_' + filehash + '.pickle'
