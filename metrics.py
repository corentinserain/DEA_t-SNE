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
    
    
    
    avg_common_neighbors = (np.mean(common_neighbors)/k)*100

    return avg_common_neighbors




def std_ratios(X_initial,X_reduced):

    distances_initial = euclidean_distances(X_initial)  #Je calcule la distance euclidienne entre chaque paire de points dans l'espace initial

    distances_reduced = euclidean_distances(X_reduced)  #Je calcule la distance euclidienne entre chaque paire de points dans l'espace réduit

    ratio = distances_reduced / (distances_initial + 0.00000000001) # au cas ou il y a une valeur = 0 meme si je pense que c'est impossoble

    std_ratios = np.std(ratio)

    return std_ratios
