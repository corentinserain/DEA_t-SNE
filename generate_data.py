from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import pandas as pd 
import os
import numpy as np


def scaler(dataset):
    """ization between 0 and 1 of data. 

    Args:
        dataset (numpy.ndarray): 

    Returns:
        numpy.ndarray: 
    """
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    return scaler.transform(dataset)


def data_generation_mc(file, samples, features, classes, informative):
    """_summary_

    Args:
        file (str): 
        samples (list): Values for the number of samples to generate
        features (list): Values for the number of features to generate
        classes (list): 
        informative (list):
    """
    all_parameters = list(product(samples, features, classes, informative))
    count_norm = 0
    count_cluster = 0
    if not os.path.exists(file):
        os.makedirs(file)
    for n_samples, n_features, n_classes, p_informative in all_parameters : 
        features, classes = make_classification(
            n_samples = n_samples, 
            n_features = n_features, 
            n_classes = n_classes, 
            n_informative = int(p_informative * n_features),
            n_redundant = 0,
            n_repeated = 0)
        dataset = pd.DataFrame(scaler(features))
        dataset['class'] = classes
        if n_classes == 1 : 
            distr = "norm"
            count_norm += 1
        else : 
            distr = "cluster"
            count_cluster += 1
        dataset.to_csv(f'./{file}/{distr}_{n_samples}_{n_features}_{n_classes}_{int(p_informative*100)}.csv')
    print(f'{count_norm} CSV files have been created with data following a normal distribution.')
    print(f'{count_cluster} CSV files have been created with data following a cluster distribution.')
        

def data_generation_uni(file, samples, features):
    """_summary_

    Args:
        file (str): 
        samples (list): Values for the number of samples to generate
        features (list): Values for the number of features to generate
    """
    all_parameters = list(product(samples, features))
    count = 0
    if not os.path.exists(file):
        os.makedirs(file)
    for n_samples, n_features in all_parameters : 
        count += 1
        dataset = pd.DataFrame(np.random.uniform(size=(n_samples, n_features)))
        dataset.to_csv(f'./{file}/uni_{n_samples}_{n_features}.csv')
    print(f'{count} CSV files have been created with data following a uniform distribution.')
        



samples = [100, 250, 500, 750, 1000, 2500, 5000]
features = [10, 20, 30, 40, 50]
classes = [1, 2, 3, 4, 5, 8, 10, 15]
informative = [0.5, 0.75, 1]
file = "data"

data_generation_mc(file, samples, features, classes, informative)
data_generation_uni(file, samples, features)

