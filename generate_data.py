from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import pandas as pd 
import os
import numpy as np


def scaler(dataset):
    """Normalization between 0 and 1 of data, using MinMaxScaler function.

    Args:
        dataset (numpy.ndarray): Contains several data to be normalized.

    Returns:
        numpy.ndarray: Contains normalized data between 0 and 1.
    """
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    return scaler.transform(dataset)


def data_generation_mc(folder, rep, samples, features, classes, informative):
    """Generating of random data with the make_classification function with all given parameter combinations and creating folders in CSV format.
    When the number of classes is equal to 1, the data follows a normal distribution. 
    When the number of classes is higher than 1, the data are separated into clusters. 
    The name of the CSV folders created corresponds to the following form: 
        distribution_samples_features_classes_(p_informative*100)_rep.csv

    Args:
        folder (str): The name of the folder where the CSV folders are generated.
        rep (int):
        samples (list): Values for the number of samples to generate.
        features (list): Values for the number of features to generate.
        classes (list): Values defining the number of classes.
        informative (list): Proportion of features that are informative.
    """
    all_parameters = list(product(samples, features, classes, informative))
    count_norm = 0
    count_cluster = 0
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(rep) : 
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
                if not os.path.exists(f'{folder}/norm'):
                    os.makedirs(f'{folder}/norm')
                count_norm += 1
                dataset.to_csv(f'{folder}/norm/{n_samples}_{n_features}_{n_classes}_{int(p_informative*100)}_{i}.csv')
            else : 
                if not os.path.exists(f'{folder}/cluster'):
                    os.makedirs(f'{folder}/cluster')
                count_cluster += 1
                dataset.to_csv(f'{folder}/cluster/{n_samples}_{n_features}_{n_classes}_{int(p_informative*100)}_{i}.csv')
            
    print(f'{count_norm} CSV files have been created with data following a normal distribution.')
    print(f'{count_cluster} CSV files have been created with data following a cluster distribution.')
        

def data_generation_uni(folder, rep, samples, features):
    """Generating of random data with the make_classification function with all given parameter combinations and creating folders in CSV format.
    The data follows a uniform distribution.
    The name of the CSV folders created corresponds to the following form: 
        distribution_samples_features_rep.csv

    Args:
        folder (str): The name of the folder where the CSV folders are generated.
        rep (int):
        samples (list): Values for the number of samples to generate.
        features (list): Values for the number of features to generate.
    """
    all_parameters = list(product(samples, features))
    count = 0
    if not os.path.exists(f'{folder}/uni'):
        os.makedirs(f'{folder}/uni')
    for i in range(rep) : 
        for n_samples, n_features in all_parameters : 
            count += 1
            dataset = pd.DataFrame(np.random.uniform(size=(n_samples, n_features)))
            dataset.to_csv(f'{folder}/uni/{n_samples}_{n_features}_{i}.csv')
    print(f'{count} CSV files have been created with data following a uniform distribution.')
        





