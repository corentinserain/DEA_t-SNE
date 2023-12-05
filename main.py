from generate_data import data_generation_mc, data_generation_uni
from tsne import tsne_metrics



#Generation des données
samples = [100, 250, 500, 750, 1000, 2500, 5000]
features = [10, 20, 30, 40, 50]
classes = [1, 2, 3, 4, 5, 8, 10, 12, 15]
informative = [0.5, 0.75, 1]
folder = "./data"

data_norm, data_cluster = data_generation_mc(folder, 5, samples, features, classes, informative)
data_uni = data_generation_uni(folder, 5, samples, features)


#t-sne
perplexity = [10, 30]
n_iter = [300, 500]

t_sne_norm = tsne_metrics(data_norm,"norm",perplexity, n_iter, False)
t_sne_uni = tsne_metrics(data_uni, "uni", perplexity, n_iter, False)
t_sne_cluster = tsne_metrics(data_cluster,"cluster",perplexity, n_iter, True)



