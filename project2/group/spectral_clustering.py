#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:11:41 2017

@author: ratnamogan
"""



# For Kmeans algorithm. See PGM otherwise for handcrafted class
from sklearn.cluster import KMeans
# For loading matlab matrix file
from scipy.io import loadmat
# Hungarian algorithm
from scipy.optimize import linear_sum_assignment

from scipy.sparse.linalg import eigs, eigsh
from scipy.stats import itemfreq
import numpy as np

from error_evaluation import evaluate_error

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})

# DATA_DIR = "./"
#
# YaleB = loadmat(DATA_DIR + 'ExtendedYaleB.mat')
# data = YaleB['EYALEB_DATA']
# data = np.array(data,dtype=np.int64)
#
#
# ground_truth = YaleB['EYALEB_LABEL'].reshape(-1) - 1
#
# keys = [2, 10, 20, 30, 38]
# data_set = {key : {"data" : data[:, ground_truth < key],
#                    "labels" : ground_truth[:(ground_truth < key).sum()]
#                   } for key in keys}
#
# data = data_set[2]['data']
# ground_truth = data_set[2]['labels']

def minWeightBipartiteMatching(clusteringA, clusteringB):
    """
    labels from cluster A will be matched on the labels from cluster B
    source : https://www.r-bloggers.com/matching-clustering-solutions-using-the-hungarian-method/
    """
    # Reshape to have column vectors
    clusteringA = clusteringA.reshape(-1)
    clusteringB = clusteringB.reshape(-1)

    # Distinct cluster ids in A and B
    idsA, idsB = np.unique(clusteringA), np.unique(clusteringB)
    # Number of instances in A and B
    nA, nB = len(clusteringA), len(clusteringB)

    if len(idsA) != len(idsB):
        raise ValueError("Dimensions of clustering do no match")
    if  nA != nB:
        raise ValueError("Lengths of clustering do no match")

    nC = len(idsA)
    tupel = np.arange(nA)

    # Computing the distance matrix
    assignmentMatrix = -1 + np.zeros((nC, nC))
    for i in range(nC):
        tupelClusterI = tupel[clusteringA == i]
        for j in range(nC):
            nA_I = len(tupelClusterI)
            tupelB_I = tupel[clusteringB == j]
            nB_I = len(tupelB_I)
            nTupelIntersect = len(np.intersect1d(tupelClusterI, tupelB_I))
            assignmentMatrix[i, j] = (nA_I - nTupelIntersect) + (nB_I - nTupelIntersect)

    # Optimization
    _, result = linear_sum_assignment(assignmentMatrix)
    return result



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
        distance_matrix = np.array([np.linalg.norm(X - x.reshape(-1, 1), axis = 0)\
                                    for x in X.T])

        # No idea why D is not symetric
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)


    np.fill_diagonal(distance_matrix, np.inf)
    W = np.inf * np.ones(distance_matrix.shape)
    # Matrix of rank of nearest neighbors
    KNN = distance_matrix.argsort()[:, :k]
    for k,i in enumerate(KNN):
        if k in KNN[i]:
            W[k, i] = distance_matrix[k, i]

    W = (W + W.T)/2

    return np.exp(-0.5 * (W / sigma) **2)

def SC(W, n, method = "full"):
    """
    Spectral Clustering algorithm.
    :param W: affinity matrix NxN (N : number of points)
    :param n: number of clusters
    :return : Segmentation of the data in n groups
    """
    # 1. Construct an affinity graph G with weight matrix W
    # 2. Compute the degree matrix D = diag(W1) and the Laplacian L = D - W
    D = np.diag(W.sum(axis = 1))
    L = D - W
    # 3. Compute the n eigenvectors of L associated with its n smallest eigenvalues
    ## May use scipy.sparse.linalg.eigs to speed up computation
    ## The vectors returned by linalg are normalized
    if method == "sparse":
        # SC(W, 10, method = "sparse")
        # 2 s ± 87.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        ## !!! Not functionning because not returning THE smallest ones
        evalues, evectors = eigsh(L, k=n, which = 'SM')
    else :
        # SC(W, 10, method = "full")
        # 37.5 s ± 2.61 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
        evalues, evectors = np.linalg.eigh(L)
    # 4. Y : transpose of matrix of eigenvectors normalized by euclidien norm
    ## Y.shape = (N x n).T => n x N
    #row_sums = evectors.sum(axis=1)
    #normalize_eigv = evectors / row_sums[:, np.newaxis]
    Y = evectors[:, :n].T
    # 5. Cluster the points {y_j}_1^N into n groups using the K-means algorithm
    ## n_jobs controls the number of threads
    ## init with random as in algorithm 4.4
    kmeans = KMeans(n_clusters = n, init = 'random').fit(Y.T)
    # Return the label for each point : not the exact segmentation as in algo 4.4
    return kmeans.labels_

# # Distance matrix
# distance_matrix = np.array([np.linalg.norm(data - x.reshape(-1, 1), axis = 0) for x in data.T])
# distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
#
#
# W = gaussian_affinity(data, 10, 1000, distance_matrix = distance_matrix)
# print ("Percentage of W filled : {}%".format((W > 0).sum() / len(W)**2 * 100))
# res = SC(W, 2)
#
# error=evaluate_error(res,ground_truth)
