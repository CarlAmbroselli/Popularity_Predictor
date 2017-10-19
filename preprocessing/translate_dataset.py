from translator import Translator
import pandas as pd
import numpy as np
from time import sleep
import random
import progressbar

translate = Translator()
comments = pd.read_csv('../data/datasets/YNACC-Evaluation/train/comments.csv')
bar = progressbar.ProgressBar(max_value=comments.shape[0], redirect_stdout=True)
i = 0

def translate_slow(text, i, bar):
  sleep(random.randint(2,15)/10)
  translated = Translator.translate(text)
  bar.update(i)
  print(translated + '\n\n')
  return translated

comments['text_de'] = comments.apply(lambda x: translate_slow(x['text'], x.name, bar), axis=1)
comments.to_csv('../data/datasets/YNACC-Evaluation/train/comments_de.csv')
