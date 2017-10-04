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
    test_df = pd.read_csv('data/datasets/' + dataset + '/test/articles.csv', sep=',')
    train_df['has_comments'] = train_df['comment_count'] > 0
    test_df['has_comments'] = test_df['comment_count'] > 0
    return (train_df, test_df)

def execute_individual(dataset='tiny'):
    print("Using dataset", dataset)
    print("Load Data...")
    train_df, test_df = load_data(dataset)
    predictor = Predictor()
    predictor.set_target('comment_count', useRegression=True)
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
    train_df, test_df = load_data(dataset)
    predictor = Predictor()
    # predictor.set_target('has_comments', useRegression=False)
    predictor.set_target('comment_count', useRegression=True)
    print("Fit...")
    predictor.fit(train_df)
    print("Predict...")
    result = predictor.predict(test_df)
    result['real'] = predictor.ground_truth(test_df)
    print("Result:")
    print(result.head(5))
    print("Metrics:")
    print(json.dumps(predictor.metrics(), indent=2))
    # visualizer = Visualize()
    # visualizer.plot_results(test_df['comment_count'], result)


def main():
    init_nltk()
    datasets = [
        'Tiny',
        # 'Tr16Te17-Small',
        # 'Tr16Te17',
        # 'Tr09-16Te17'
    ]
    for dataset in datasets:
        # execute(dataset)
        execute_individual(dataset)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()