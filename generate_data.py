from sklearn.datasets import make_classification
from itertools import product
import pandas as pd 
import os

def parameters_mc() : 
    n_samples = [100,200,300,400,500]
    n_features = [10,20,30,40,50]
    n_classes = [2,3,4,5]
    # ajouter d'autres paramètres
    all_parameters = list(product(n_samples, n_features, n_classes))
    return all_parameters

def data_generation_mc():
    all_parameters = parameters_mc()
    for n_samples, n_features, n_classes in all_parameters : 
        features, labels = make_classification(n_samples = n_samples, n_features = n_features, n_classes = n_classes, n_informative = n_classes)
        dataset = pd.DataFrame(features)
        dataset['label'] = labels
        dataset.to_csv(f'./data/data_{n_samples}_{n_features}_{n_classes}.csv')

if not os.path.exists('data'):
    os.makedirs('data')
data_generation_mc()

