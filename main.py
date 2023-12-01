from generate_data import data_generation_mc, data_generation_uni
from tsne import tsne_all

#Generation des données
samples = [100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000]
features = [10, 20, 30, 40, 50, 80]
classes = [1, 2, 3, 4, 5, 8, 10, 12, 15]
informative = [0.5, 0.75, 1]
folder = "./data"

data_generation_mc(folder, 5, samples, features, classes, informative)
data_generation_uni(folder, 5, samples, features)

#t-sne
perplexity = [10, 30, 50]
n_iter = [300, 500, 1000]
distr = ["uni", "cluster", "norm"]
for i in distr :
    input_folder = f"./data/{i}"
    output_folder = f"./t_sne/{i}"
    tsne_all(input_folder, output_folder, perplexity, n_iter)
    print("T-sne was successfully applied to the data files with the distribution ", i)
