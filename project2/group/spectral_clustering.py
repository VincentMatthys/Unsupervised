#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For Kmeans algorithm
from sklearn.cluster import KMeans
import numpy as np

def gaussian_affinity(X, k, sigma, distance_matrix = None):
    """
    Construction of gaussian affinity with K-NN :
    w_{ij} = exp(-d_{ij}^2 / 2s^2) if NN 0 else

    Parameters :
    ------------
    X: array, shape [NxD]
       N data points
    k: positive integer
       number of nearest neighbors to consider
    sigma: positive integer
       standard deviation of the gaussian kernel

    Returns :
    ---------
    W: array, shape [NxN]
       affinity matrix

    """
    if distance_matrix is None:
#         # Care not to swap with this faster method (minimum RAM for YaleB : 32Gb)
#         A = np.tile(X, (X.shape[1], 1, 1)) - X.T.reshape(X.shape[1], -1, 1)
#         D = np.linalg.norm(A, axis = 1)

        # Awful method
        distance_matrix = np.array([\
        np.linalg.norm(X - x.reshape(-1, 1), axis = 0) for x in X.T])

        # To force the matrix to be symmetric
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)


    np.fill_diagonal(distance_matrix, np.inf)
    W = np.inf * np.ones(distance_matrix.shape)
    # Matrix of rank of nearest neighbors
    KNN = distance_matrix.argsort()[:, :k]
    for k,i in enumerate(KNN):
        if k in KNN[i]:
            W[k, i] = distance_matrix[k, i]

    W = (W + W.T)/2

    return np.exp(-0.5 * (W / sigma) ** 2)

def SC(W, n):
    """
    Spectral Clustering

    Inspired from : algorithm 6.1
    Vidal, René, Yi Ma, and S. Shankar Sastry. "Principal Component Analysis." Generalized Principal Component Analysis. Springer New York, 2016

    Parameters:
    -----------
    W:      array, shape [N, N]
            affinity matrix NxN (N : number of points)
    n:      positive integer
            number of clusters

    Returns:
    --------
    groups:  array-like, shape (N,)
             Segmentation of the data in n groups,
             labels (between 0 and n-1) for each point
    """
    # 1. Compute the degree matrix D = diag(W1) and the Laplacian L = D - W
    D = np.diag(W.sum(axis = 1))
    L = D - W
    # 2. Compute the n eigenvectors of L associated with its n smallest eigenvalues
    ## The vectors returned by linalg are normalized
    _, evectors = np.linalg.eigh(L)
    # 3. Cluster the points {y_j}_1^N into n groups using the K-means algorithm
    ## n_jobs controls the number of threads
    ## init with random as in algorithm 4.4
    kmeans = KMeans(n_clusters = n, init = 'random').fit(evectors[:, :n])
    # Return the label for each point
    return kmeans.labels_
