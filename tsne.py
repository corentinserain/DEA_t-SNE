import os
import sklearn
import pandas as pd
from itertools import product
from sklearn.manifold import TSNE 
from generate_data import scaler
from metrics import average_common_neighbors, std_ratios

def tsne(df, folder, perplexity, n_iter, classe):
    """Apply the tsne algorithm on a single dataframe with all combinations of parameters (perplexity, number of iterations) 
    and create a csv file with the tsne result for each combination. 
    The name of the CSV folders created corresponds to the following form: 
        perplexity_n_iter_samples_features_classes_(p_informative*100)_rep.csv

    Args:
        df (DataFrame): Data for which applied tsne.
        folder (str): The name of the folder where the CSV files are generated.
        perplexity (list): Values for the tsne parameter perplexity.
        n_iter (list): Values for the tsne parameter n_iter.
        classe (bool): When True, the dataframe contains a class column. 

    Returns:
        tsne_results (list): All tsne results generated from the same data file.
    """
    tsne_results = []
    all_parameters = list(product(perplexity,n_iter))
    df_name = df.name
    if not os.path.exists(folder):
        os.makedirs(folder)

    if classe :
        classes = df['class']
        df = df.drop('class', axis=1) 

    for perplexity_val, n_iter_val in all_parameters:
        name = f'{perplexity_val}_{n_iter_val}_{df_name}'
        tsne = sklearn.manifold.TSNE(perplexity = perplexity_val, n_iter = n_iter_val)
        tsne_result = pd.DataFrame(scaler(tsne.fit_transform(df)))
        if classe :
            tsne_result['class'] = classes

        tsne_result.to_csv(f'{folder}/{name}.csv')

        tsne_result.name = name
        tsne_results.append(tsne_result)

    return tsne_results


def tsne_metrics(data, distr, perplexity, n_iter, classe):
    """Apply the tsne algorithm to all data files for a distribution, with all combinations of parameters (perplexity, number of iterations). 
    Calculate two metrics(average common neighbors and ratios) based on the data file and its associated tsne result files. 
    Create a csv file with a row for the metric results and parameters for each tsne result.

    Args:
        data (list): Data file names.
        distr (str): Data distribution.
        perplexity (list): Values for the tsne parameter perplexity.
        n_iter (list): Values for the tsne parameter n_iter.
        classe (bool): When True, the dataframe contains a class column. 

    Returns:
        df_metric (DataFrame): Metric results and parameters for each tsne result.
    """
    if distr == "uni":
        columns = ["perplexity", "n_iter", "n_samples", "n_features", "avg_common_neighbors10", "avg_common_neighbors40", "std_ratios"]
    else : 
        columns = ["perplexity", "n_iter", "n_samples", "n_features", "n_classes", "p_informative", "avg_common_neighbors10", "avg_common_neighbors40", "std_ratios"]
    
    df_metric = pd.DataFrame(columns=columns)
    line = 1

    for file_name in data:
        df_data = pd.read_csv(f'./data/{distr}/{file_name}.csv')
        df_data.name = file_name
        dfs_t_sne = tsne(df_data, f'./t_sne/{distr}', perplexity, n_iter, classe)

        for df_t_sne in dfs_t_sne : 
            name = df_t_sne.name
            param = name.split("_")
            del(param[-1])

            if classe :
                df_t_sne = df_t_sne.drop('class', axis=1) 

            param.append(average_common_neighbors(df_data, df_t_sne, 10))
            param.append(average_common_neighbors(df_data, df_t_sne, 40))

            param.append(std_ratios(df_data, df_t_sne))
            
            df_metric.loc[line]=param
            line += 1
            
    df_metric.to_csv(f'metrics_results_{distr}.csv')
    return df_metric


    



