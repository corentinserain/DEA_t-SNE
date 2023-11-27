from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import pandas as pd 
import os
import numpy as np

def parameters_mc(): 
    n_samples = [100,200,300]
    n_features = [10,20,30,40,50]
    n_classes = [2,3,4,5]
    p_informative = [0.5, 0.75, 1]
    all_parameters = list(product(n_samples, n_features, n_classes, p_informative))
    return all_parameters

def parameters_uni():
    

def scaler(dataset):
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    return scaler.transform(dataset)

def data_generation_mc():
    all_parameters = parameters_mc()
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
        dataset.to_csv(f'./data/samples{n_samples}_features{n_features}_classes{n_classes}_informative{int(p_informative*100)}.csv')


if not os.path.exists('data'):
    os.makedirs('data')
data_generation_mc()

