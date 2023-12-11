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
        samples_features_classes_(p_informative*100)_rep.csv

    Args:
        folder (str): The name of the folder where the CSV files are generated.
        rep (int): Number of repetitions of data generation.
        samples (list): Values for the number of samples to generate.
        features (list): Values for the number of features to generate.
        classes (list): Values defining the number of classes.
        informative (list): Proportion of features that are informative.

    Returns:
        list_norm (list): Names of CSV files created for normal distribution.
        list_cluster (list): Names of CSV files created for cluster distribution.
    """
    all_parameters = list(product(samples, features, classes, informative))
    list_norm = []
    list_cluster = []
    if not os.path.exists(f'{folder}/norm'):
        os.makedirs(f'{folder}/norm')
    if not os.path.exists(f'{folder}/cluster'):
        os.makedirs(f'{folder}/cluster')
    
    for i in range(rep) : 
        for n_samples, n_features, n_classes, p_informative in all_parameters : 
            name = f'{n_samples}_{n_features}_{n_classes}_{int(p_informative*100)}_{i}'
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
                dataset.to_csv(f'{folder}/norm/{name}.csv')   
                list_norm.append(name)

            else : 
                dataset.to_csv(f'{folder}/cluster/{name}.csv')
                list_cluster.append(name)
 
    print(f'{len(list_norm)} CSV files have been created with data following a normal distribution.')
    print(f'{len(list_cluster)} CSV files have been created with data following a cluster distribution.')
    return list_norm, list_cluster
        

def data_generation_uni(folder, rep, samples, features):
    """Generating of random data with the random uniform function with all given parameter combinations and creating folders in CSV format.
    The data follows a uniform distribution.
    The name of the CSV folders created corresponds to the following form: 
        samples_features_rep.csv

    Args:
        folder (str): The name of the folder where the CSV folders are generated.
        rep (int): Number of repetitions of data generation.
        samples (list): Values for the number of samples to generate.
        features (list): Values for the number of features to generate.

    Returns:
        list_uni (list): Names of CSV files created for uniform distribution.
    """
    all_parameters = list(product(samples, features))
    list_uni = []
    if not os.path.exists(f'{folder}/uni'):
        os.makedirs(f'{folder}/uni')

    for i in range(rep) : 
        for n_samples, n_features in all_parameters : 
            name = f'{n_samples}_{n_features}_{i}'
            dataset = pd.DataFrame(np.random.uniform(size=(n_samples, n_features)))
            dataset.to_csv(f'{folder}/uni/{name}.csv')
            list_uni.append(name)

    print(f'{len(list_uni)} CSV files have been created with data following a uniform distribution.')
    return list_uni


        





