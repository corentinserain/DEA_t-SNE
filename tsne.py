from sklearn.manifold import TSNE
import sklearn
import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from generate_data import scaler
import os


def tsne(df, output_folder, perplexity, n_iter, classe):
    if classe :
        y = df['class']
        X = df.drop('class', axis=1) 
    else : X = df
    all_parameters = list(product(perplexity,n_iter))
    for perplexity_val, n_iter_val in all_parameters:
        tsne = sklearn.manifold.TSNE(perplexity = perplexity_val, n_iter = n_iter_val)
        tsne_result = pd.DataFrame(scaler(tsne.fit_transform(X)))
        if classe :
            tsne_result['class'] = y

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        tsne_result.to_csv(f'{output_folder}/{perplexity_val}_{n_iter_val}-{df.name}.csv')
        print(f'{output_folder}/{perplexity_val}_{n_iter_val}-{df.name}.csv')
        tsne_result.name = f'{perplexity_val}_{n_iter_val}-{df.name}'
    return tsne_result

def tsne_all(input, output_folder, perplexity, n_iter, classe):
    liste = []
    for df in input:
        result = tsne(df, output_folder, perplexity, n_iter, classe)
        liste.append(result)
    return liste


    



