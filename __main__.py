import pandas as pd
from predictor import Predictor
from analysis.visualize import Visualize
import numpy as np
import nltk
import os
import json

def init_nltk():
    if not os.path.exists('nltk'):
        os.makedirs('nltk')
    nltk.data.path.append(os.getcwd() + '/nltk')
    dependencies = ['corpora/stopwords']
    # dependencies = ['corpora/stopwords', 'maxent_ne_chunker','words', 'punkt', 'averaged_perceptron_tagger', 'tokenizers/punkt/PY3/english.pickle']
    for package in dependencies:
        try:
            nltk.data.find(package)
        except LookupError:
            nltk.download(package, os.getcwd() + '/nltk')

def load_data(dataset='tiny'):
    train_df = pd.read_csv('data/datasets/' + dataset + '/train/articles.csv', sep=',')
    train_df = train_df[train_df['year'] >= 2014]
    test_df = pd.read_csv('data/datasets/' + dataset + '/test/articles.csv', sep=',')
    test_df = test_df[test_df['year'] >= 2014]
    # train_df = pd.read_csv('data/datasets/' + dataset + '/train/comments.csv', sep=',')
    # test_df = pd.read_csv('data/datasets/' + dataset + '/test/comments.csv', sep=',')
    # train_df['text'] = train_df['text_de']
    # test_df['text'] = test_df['text_de']
    # train_df = train_df[train_df['ressort'] == 'karriere']
    # test_df = test_df[test_df['ressort'] == 'karriere']
    train_df['no_comments'] = train_df['comment_count'] == 0
    test_df['no_comments'] = test_df['comment_count'] == 0
    train_df['has_comments'] = train_df['comment_count'] > 0
    test_df['has_comments'] = test_df['comment_count'] > 0
    high_volume = np.percentile(train_df['comment_count'], 90)
    print("High Volume = {}".format(high_volume))
    train_df['has_many_comments'] = train_df['comment_count'] > high_volume
    test_df['has_many_comments'] = test_df['comment_count'] > high_volume
    return (train_df, test_df)

def execute(dataset='tiny', individual=False):
    print("Using dataset", dataset)
    print("Load Data...")

    # napoles
    # targets = [('y_persuasive', 'classification'), ('y_audience', 'classification'), ('y_agreement_with_commenter', 'classification'), ('y_informative', 'classification'), ('y_mean', 'classification'), ('y_controversial', 'classification'), ('y_disagreement_with_commenter', 'classification'), ('y_off_topic_with_article', 'classification'), ('y_sentiment_neutral', 'classification'), ('y_sentiment_positive', 'classification'), ('y_sentiment_negative', 'classification'), ('y_sentiment_mixed', 'classification')]

    # tsagkias
    # targets = [('no_comments', 'classification'), ('has_comments', 'classification'), ('has_many_comments', 'classification'), ('comment_count', 'regression'), ('facebook_shares', 'regression')]
    targets = [('no_comments', 'regression'), ('has_comments', 'regression'), ('has_many_comments', 'regression'), ('no_comments', 'classification'), ('has_comments', 'classification'), ('has_many_comments', 'classification'), ('comment_count', 'regression'), ('facebook_shares', 'regression'), ('comment_count_minus_last_month_average', 'regression'), ('facebook_shares_minus_last_month_average', 'regression'), ('shared_on_facebook', 'classification'), ('likes_on_facebook', 'regression'), ('comments_on_facebook', 'regression')]
    # targets = [('comment_count_minus_last_month_average', 'regression'), ('facebook_shares_minus_last_month_average', 'regression'), ('shared_on_facebook', 'classification'), ('likes_on_facebook', 'regression'), ('comments_on_facebook', 'regression')]


    # bandari
    # targets = [('facebook_shares', 'regression')]

    train_df, test_df = load_data(dataset)
    for target in targets:
        print('=' * 50)
        print(target[0], target[1])
        print('*' * 50)
        print('=' * 50)
        predictor = Predictor()
        predictor.set_target(target[0], useRegression=(target[1] == 'regression'))
        print("Fit all features...")
        predictor.fit(train_df)
        print("Predict...")
        result = predictor.predict(test_df)
        result['real'] = predictor.ground_truth(test_df)
        # print("Result:")
        # print(result.head(5))
        print("Metrics for {}:".format(target))
        print(json.dumps(predictor.metrics(), indent=2))
        try:
            visualizer = Visualize()
            visualizer.plot_roc(predictor.ground_truth(test_df), result['logistic regression'])
        except:
            pass
        # visualizer.plot_roc(predictor.ground_truth(test_df), result['linear_regression'])
        if individual:
            features = predictor.features
            for feature in features:
                print("Feature: {}".format(feature[0]))
                predictor.features = [feature]
                predictor.fit(train_df)
                result = predictor.predict(test_df)
                result['real'] = predictor.ground_truth(test_df)
                print(json.dumps(predictor.metrics(), indent=2))

def main():
    print("Init")
    init_nltk()
    datasets = [
        # 'Tiny',
        # 'YNACC-Evaluation',
        # 'YNACC',
        # 'Tr16Te17-Small',
        # 'Tr16Te17',
        # 'Tr09-16Te17',
        'Tr14-16Te17'
    ]
    print("Execute")
    for dataset in datasets:
        execute(dataset, individual=True)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
