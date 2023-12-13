import sklearn
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


def average_common_neighbors(X_initial, X_reduced, k):
    """Calcule le nombre moyen de k voisins en communs entre un espace initial et un espace réduit.

    Args:
        X_initial (DataFrame): Espace initial
        X_reduced (DataFrame): Espace réduit
        k (int): nombre de voisins

    Returns:
        float: pourcentage de voisins en communs
    """
    knn_initial = NearestNeighbors(n_neighbors=k)
    knn_initial.fit(X_initial)
    distances, indices_initial = knn_initial.kneighbors(X_initial)         
    
    knn_reduced = NearestNeighbors(n_neighbors=k)
    knn_reduced.fit(X_reduced)
    distances, indices_reduced = knn_reduced.kneighbors(X_reduced)
    
    common_neighbors = []
    for i in range(len(indices_initial)):
        neighbors_initial = set(indices_initial[i]) 
        neighbors_reduced = set(indices_reduced[i])  
        common_neighbors.append(len(neighbors_initial.intersection(neighbors_reduced)))
    
    avg_common_neighbors = (np.mean(common_neighbors)/k)*100

    return avg_common_neighbors


def std_ratios(X_initial, X_reduced):
    """Calcule l'écart type des ratios de distances euclidiennes entre les points dans l'espace initial et l'espace réduit.

    Args:
        X_initial (DataFrame): Espace initial
        X_reduced (DataFrame): Espace réduit

    Returns:
        float: Ecart-type du ratio
    """
    distances_initial = euclidean_distances(X_initial)  
    distances_reduced = euclidean_distances(X_reduced)  
    ratio = distances_reduced / (distances_initial + 0.00000000001) 
    std_ratios = np.std(ratio)

    return std_ratios
