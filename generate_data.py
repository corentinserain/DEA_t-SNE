from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import pandas as pd 
import os
import numpy as np

def parameters_mc(samples, features): 
    n_classes = [2,3,4,5]
    p_informative = [0.5, 0.75, 1]
    all_parameters = list(product(samples, features, n_classes, p_informative))
    return all_parameters

def parameters_uni(samples, features):
    all_parameters = list(product(samples, features))
    return all_parameters

def scaler(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    return scaler.transform(dataset)

def data_generation_mc(samples, features):
    all_parameters = parameters_mc(samples, features)
    count = 0
    for n_samples, n_features, n_classes, p_informative in all_parameters : 
        count += 1
        features, classes = make_classification(
            n_samples = n_samples, 
            n_features = n_features, 
            n_classes = n_classes, 
            n_informative = int(p_informative * n_features),
            n_redundant = 0,
            n_repeated = 0) 
        dataset = pd.DataFrame(scaler(features))
        dataset['class'] = classes
        dataset.to_csv(f'./data/mc_samples{n_samples}_features{n_features}_classes{n_classes}_informative{int(p_informative*100)}.csv')
    print(f'{count} CSV files have been created with data following a normal distribution.')
        
def data_generation_uni(samples, features):
    all_parameters = parameters_uni(samples, features)
    count = 0
    for n_samples, n_features in all_parameters : 
        count += 1
        dataset = pd.DataFrame(np.random.uniform(size=(n_samples, n_features)))
        dataset.to_csv(f'./data/uni_samples{n_samples}_features{n_features}.csv')
    print(f'{count} CSV files have been created with data following a uniform distribution.')
        

if not os.path.exists('data'):
    os.makedirs('data')
samples = [100,200,300,500,1000,5000,10000]
features = [10,20,30,40,50]
data_generation_mc(samples, features)
data_generation_uni(samples, features)

