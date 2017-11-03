import numpy as np
from feature.features import Features
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard
from sklearn.utils import class_weight
import pandas as pd

class CNN_Classification(Features):
  def __init__(self):
    super().__init__('cnn_classification')
    self.is_first_run = True
    self.tokenizer = None
    self.ground_truth = 'has_comments'
    self.model = None
    print('==== CNN uses {} as target variable ===='.format(self.ground_truth))

  def _extract_features(self, df):
    top_words = 50000
    max_text_length = 3000
    embedding_vecor_length = 300
    if self.is_first_run:
      self.is_first_run = False
      self.tokenizer = Tokenizer(num_words=top_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n',
                            lower=True, split=" ", char_level=False)
      self.tokenizer.fit_on_texts(df['text'])
      train_seq = self.tokenizer.texts_to_sequences(df['text'])
      y_train = df[self.ground_truth]

      c_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
      X_train = sequence.pad_sequences(train_seq, maxlen=max_text_length)
      self.model = Sequential()
      self.model.add(Embedding(top_words, embedding_vecor_length, input_length=max_text_length))

      self.model.add(Convolution1D(64, 3, padding='same'))
      self.model.add(Convolution1D(32, 3, padding='same'))
      self.model.add(Convolution1D(16, 3, padding='same'))
      self.model.add(Flatten())
      self.model.add(Dropout(0.2))
      self.model.add(Dense(180, activation='sigmoid'))
      self.model.add(Dropout(0.2))
      self.model.add(Dense(1, activation='sigmoid'))
      tensorBoardCallback = TensorBoard(write_graph=True)
      self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      self.model.fit(X_train, y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=128, class_weight=c_weight)
      scores = self.model.evaluate(X_train, y_train, verbose=1)
      print("CNN training accuracy: %.2f%%" % (scores[1] * 100))
      result = self.model.predict(X_train)
      # print(result)
      return result
    else:
      test_seq = self.tokenizer.texts_to_sequences(df['text'])
      X_test = sequence.pad_sequences(test_seq, maxlen=max_text_length)
      result = self.model.predict(X_test)
      # print(result)
      return result
