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
    train_df = pd.read_csv('data/datasets/' + dataset + '/train/comments.csv', sep=',')
    test_df = pd.read_csv('data/datasets/' + dataset + '/test/comments.csv', sep=',')
    return (train_df, test_df)

def execute_individual(dataset='tiny'):
    print("Using dataset", dataset)
    print("Load Data...")
    train_df, test_df = load_data(dataset)
    predictor = Predictor()
    # tagets = ['y_persuasive', 'y_audience', 'y_agreement_with_commenter', 'y_informative', 'y_mean', 'y_controversial', 'y_disagreement_with_commenter', 'y_off_topic_with_article', 'y_sentiment_neutral', 'y_sentiment_positive', 'y_sentiment_negative', 'y_sentiment_mixed']
    predictor.set_target('y_persuasive', useRegression=False)
    features = predictor.features
    for feature in features:
        print("Feature: {}".format(feature[0]))
        predictor.features = [feature]
        predictor.fit(train_df)
        result = predictor.predict(test_df)
        result['real'] = predictor.ground_truth(test_df)
        print(json.dumps(predictor.metrics(), indent=2))

def execute(dataset='tiny'):
    print("Using dataset", dataset)
    print("Load Data...")
    targets = ['y_persuasive', 'y_audience', 'y_agreement_with_commenter', 'y_informative', 'y_mean', 'y_controversial', 'y_disagreement_with_commenter', 'y_off_topic_with_article', 'y_sentiment_neutral', 'y_sentiment_positive', 'y_sentiment_negative', 'y_sentiment_mixed']
    train_df, test_df = load_data(dataset)
    for target in targets:
        predictor = Predictor()
        predictor.set_target(target, useRegression=False)
        print("Fit...")
        predictor.fit(train_df)
        print("Predict...")
        result = predictor.predict(test_df)
        result['real'] = predictor.ground_truth(test_df)
        print("Result:")
        print(result.head(5))
        print("Metrics for {}:".format(target))
        print(json.dumps(predictor.metrics(), indent=2))
        visualizer = Visualize()
        # visualizer.plot_roc(predictor.ground_truth(test_df), result['svr'])
        # visualizer.plot_roc(predictor.ground_truth(test_df), result['linear_regression'])


def main():
    init_nltk()
    datasets = [
        # 'Tiny',
        'YNACC-Evaluation',
        # 'YNACC',
        # 'Tr16Te17-Small',
        # 'Tr16Te17',
        # 'Tr09-16Te17'
    ]
    for dataset in datasets:
        execute(dataset)
        # execute_individual(dataset)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
