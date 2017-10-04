import hashlib
import numpy as np
import os.path
import pickle
from termcolor import colored
from glob import glob
import os


class Features:
    def __init__(self, file):
        self.file = file
        pass

    def extract_features(self, df):
        cached = self.load_cached(df)
        # self.cleanup(df)
        if cached is not None:
            return cached
        else:
            print('Recalculating:', colored(self.file, 'red', attrs=['bold']))
            features = self._extract_features(df)
            filename = self.filepath(df)[0]
            pickle.dump(features, open(filename, 'wb'))
            print('saved features at:', filename)
            return features

    def _extract_features(self, df):
        raise Exception('Unsupported Method', 'The child feature generator should overwrite this method.')

    def load_cached(self, df):
        filepath = self.filepath(df)[0]
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

        return ('feature/cache/' + hash + '_' + self.file + '_' + filehash + '.pickle', hash, self.file, filehash)

    def cleanup(self, df):
        filename, df_hash, file, filehash = self.filepath(df)
        for f in glob('feature/cache/' + df_hash + '_' + file + '_' + '*' + '.pickle'):
            if f != filename:
                os.remove(f)

    def reset(self):
        pass

