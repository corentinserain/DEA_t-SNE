from generate_data import data_generation_mc, data_generation_uni
from tsne import tsne_all
# from metrics import average_common_neighbors



#Generation des données
#samples = [100, 200, 400, 600, 800, 1000, 2500, 5000]
#samples = [100, 250, 500, 750, 1000, 2500, 5000]
samples = [100, 250, 500]
features = [10, 20, 30]
classes = [1, 2, 5, 8]
informative = [0.5, 0.75, 1]
folder = "./data"

data_norm, data_cluster = data_generation_mc(folder, 5, samples, features, classes, informative)
data_uni = data_generation_uni(folder, 5, samples, features)


#t-sne
perplexity = [10, 30, 50]
n_iter = [300, 500, 1000]

t_sne_norm = tsne_all(data_norm,"./t_sne/norm",perplexity, n_iter)
t_sne_uni = tsne_all(data_uni,"./t_sne/uni",perplexity, n_iter)
t_sne_cluster = tsne_all(data_cluster,"./t_sne/cluster",perplexity, n_iter)


#Metrics

#KNN


# for i in distr :
#     input_data = f"./data/{i}"
#     input_t_sne = f"./t_sne/{i}"
#     average_common_neighbors(input_data, input_t_sne,k=10)
#     average_common_neighbors(input_data, input_t_sne,k=40)
#     print('ca a marché!!!')