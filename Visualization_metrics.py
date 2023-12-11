import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('metrics_results_cluster.csv')


parametres = ['perplexity', 'n_iter', 'n_samples', 'n_features', 'n_classes', 'p_informative']
metriques = ['avg_common_neighbors10', 'avg_common_neighbors40','std_ratios']


for parametre in parametres:
    for metrique in metriques:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=parametre, y=metrique, data=data)
        plt.title(f'{metrique} en fonction de {parametre}')
        plt.show()