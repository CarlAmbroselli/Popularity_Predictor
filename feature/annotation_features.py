import numpy as np
import pandas as pd
from feature.features import Features
import progressbar
import code

class AnnotationFeatures(Features):
  def __init__(self):
    super().__init__('annotation_features')
    self.annotations = None
    self.result_columns = ['t_persuasive', 't_not_persuasive', 't_intended_audience', 't_broadcast_message',
                        't_offtopic_with_article', 't_topic_personal_story', 't_topic_offtopic_with_conversation',
                        't_constructive', 't_not_constructive', 't_continual_disagreement', 't_agreement_throughout',
                        't_initial_agreement_later_disagreement', 't_initial_disagreement_later_agreement',
                        't_argumentative', 't_positive_respectful', 't_offtopic_digression', 't_flamewar_insulting',
                        't_snarky_humorous', 't_personal_story', 't_negative_sentiment', 't_positive_sentiment',
                        't_neutral_sentiment', 't_mixed_sentiment', 't_tone_sympathetic', 't_tone_mean',
                        't_tone_sarcastic', 't_tone_funny', 't_tone_informative', 't_tone_controversial',
                        't_disagreement_with_commenter', 't_agreement_with_commenter', 't_adjunct_opinion']

  def first_page_comments(self, url, bar):
      if not url in self.group_keys:
          return [-1] * len(self.result_columns)

      comments = self.grouped_annotations.get_group(url)

      bar['counter'] = bar['counter'] + 1
      bar['bar'].update(bar['counter'])

      try:
          result = comments[(comments.nth_comment_in_article <= 4.0) & (comments.nth_comment_in_article % 1 == 0)]
          if len(result) == 0:
            return [-1] * len(self.result_columns)
          else:
            return np.average(result[self.result_columns], axis=0)
      except:
          code.interact(local=locals())

  def _extract_features(self, df):
    if self.annotations is None:
        self.annotations = pd.read_csv('data/comments_with_learned_annotations.csv')
        self.grouped_annotations = self.annotations.groupby(['url'])
        self.group_keys = self.grouped_annotations.groups.keys()

    bar = {}
    bar['bar'] = progressbar.ProgressBar(max_value=len(df))
    bar['counter'] = 0
    return np.vstack(df['url'].apply(lambda x: self.first_page_comments(x, bar)))