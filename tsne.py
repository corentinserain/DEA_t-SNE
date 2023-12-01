from sklearn.manifold import TSNE
import sklearn
import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from generate_data import scaler

import os



def tsne(file, output_folder, perplexity, n_iter):
    df = pd.read_csv(file, index_col=0)
    all_parameters = list(product(perplexity,n_iter))
    filename = os.path.basename(file)
    filename, _ = os.path.splitext(filename)
    for perplexity_val, n_iter_val in all_parameters:
        tsne = sklearn.manifold.TSNE(perplexity = perplexity_val, n_iter = n_iter_val)
        tsne_result = pd.DataFrame(scaler(tsne.fit_transform(df)))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        tsne_result.to_csv(f'{output_folder}/{perplexity_val}_{n_iter_val}_{filename}.csv')

def tsne_all(input_folder, output_folder, perplexity, n_iter):
    for file in os.listdir(input_folder):
        tsne(f'{input_folder}/{file}', output_folder, perplexity, n_iter)



