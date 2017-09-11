import pandas as pd
from predictor import Predictor
from analysis.visualize import Visualize
import numpy as np
import nltk
import os
import pprint

def init_nltk():
    if not os.path.exists('nltk'):
        os.makedirs('nltk')
    nltk.data.path.append(os.getcwd() + '/nltk')
    dependencies = ['corpora/stopwords']
    for package in dependencies:
        try:
            nltk.data.find(package)
        except LookupError:
            nltk.download(package, os.getcwd() + '/nltk')

def load_data(dataset='tiny'):
    train_df = pd.read_csv('data/' + dataset + '/train.csv', sep=',')
    test_df = pd.read_csv('data/' + dataset + '/test.csv', sep=',')
    train_df.dropna()
    test_df.dropna()
    return (train_df, test_df)

def execute(dataset='tiny'):
    print("Using dataset", dataset)
    print("Load Data...")
    train_df, test_df = load_data(dataset)
    predictor = Predictor()
    print("Fit...")
    predictor.fit(train_df)
    print("Predict...")
    result = predictor.predict(test_df)
    result['real'] = test_df['comment_count']
    print("Result:")
    print(result.head(20))
    print("Metrics:")
    pprint.PrettyPrinter().pprint(predictor.metrics())
    visualizer = Visualize()
    visualizer.plot_results(test_df['comment_count'], result)


def main():
    init_nltk()
    datasets = [
        # '100',
        # '1000'
        '10000',
        # '100000',
    ]
    for dataset in datasets:
        execute(dataset)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()