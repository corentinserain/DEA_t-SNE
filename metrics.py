from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import os


def average_common_neighbors(X_initial, X_reduced, k):
    # Je cherche les k plus proches voisins dans l'espace initial
    knn_initial = NearestNeighbors(n_neighbors=k)
    knn_initial.fit(X_initial)
    distances, indices_initial = knn_initial.kneighbors(X_initial)          #kneighbors renvoie les indices et les distances 
                                                                            #des voisins de chaque point.
    #print(indices_initial)
    #print(distances)
    
    # Je cherche les k plus proches voisins dans l'espace réduit
    knn_reduced = NearestNeighbors(n_neighbors=k)
    knn_reduced.fit(X_reduced)
    distances, indices_reduced = knn_reduced.kneighbors(X_reduced)
    #print(indices_reduced)
    
    # Je calcule le nombre moyen de voisins en commun
    common_neighbors = []

    for i in range(len(indices_initial)):
        neighbors_initial = set(indices_initial[i]) #j'utilise set parce qu'avec les listes c'etait long pour trouver les indices en commun. Set transforme la liste
        neighbors_reduced = set(indices_reduced[i])  # en un ensemble de données
        common_neighbors.append(len(neighbors_initial.intersection(neighbors_reduced)))
    
    
    print(len(neighbors_reduced))
    
    avg_common_neighbors = (np.mean(common_neighbors)/k)*100

    return avg_common_neighbors


def metrics_all(input_data,input_t_sne):
    for file in os.listdir(input):
        average_common_neighbors











"""Utilisation de la fonction
avg_common_neighbors10 = average_common_neighbors(X, X_tsne, k=10)
avg_common_neighbors40 = average_common_neighbors(X, X_tsne, k=40)
print("Le pourcentage moyen de voisins en commun en utilisant k=10 voisins est" ,avg_common_neighbors10, "%")
print("Le pourcentage moyen de voisins en commun en utilisant k=40 voisins est" ,avg_common_neighbors40, "%")"""