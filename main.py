from generate_data import data_generation_mc, data_generation_uni

samples = [100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000]
features = [10, 20, 30, 40, 50, 80]
classes = [1, 2, 3, 4, 5, 8, 10, 12, 15]
informative = [0.5, 0.75, 1]
folder = "../../../espaces/travail/Projet_DEA"

data_generation_mc(folder, 5, samples, features, classes, informative)
data_generation_uni(folder, 5, samples, features)