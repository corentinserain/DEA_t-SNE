from sklearn.manifold import TSNE
import sklearn
import pandas as pd
from itertools import product
from generate_data import scaler
from metrics import average_common_neighbors, std_ratios
import os



def tsne(df, output_folder, perplexity, n_iter, classe):
    tsne_results = []
    if classe :
        y = df['class']
        X = df.drop('class', axis=1) 
    else : X = df
    all_parameters = list(product(perplexity,n_iter))
    for perplexity_val, n_iter_val in all_parameters:
        name = f'{perplexity_val}_{n_iter_val}_{df.name}'
        tsne = sklearn.manifold.TSNE(perplexity = perplexity_val, n_iter = n_iter_val)
        tsne_result = pd.DataFrame(scaler(tsne.fit_transform(X)))
        if classe :
            tsne_result['class'] = y

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        tsne_result.to_csv(f'{output_folder}/{name}.csv')
        tsne_result.name = name
        tsne_results.append(tsne_result)
    return tsne_results

def tsne_metrics(data_liste, distr, perplexity, n_iter, classe):
    if distr == "uni":
        columns = ["perplexity", "n_iter", "n_samples", "n_features", "avg_common_neighbors10", "avg_common_neighbors40", "std_ratios"]
    else : 
        columns = ["perplexity", "n_iter", "n_samples", "n_features", "n_classes", "p_informative", "avg_common_neighbors10", "avg_common_neighbors40", "std_ratios"]
    df_metric = pd.DataFrame(columns=columns)
    count = 1
    for file_name in data_liste:
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
            
            df_metric.loc[count]=param
            count += 1
            
    df_metric.to_csv(f'metrics_results_{distr}.csv')
    return df_metric


    



