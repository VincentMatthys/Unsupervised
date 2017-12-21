# -*- coding: utf-8 -*-
import numpy as np
import scipy
from sklearn.cluster import KMeans


def soft_thresholding(mat, threshold):
    """
    Soft thresholding operator
    Proximal operator of the L1 norm

    Parameters:
    -----------
    :param mat:     array-like
                    matrix to threshold
    :param threshold: real value
                    value of threshold

    Returns:
    --------
    mat:            array-like
                    thresholded matrix
    """

    mat[np.abs(mat) < threshold] = 0
    mat[mat > threshold] -= threshold
    mat[mat < -threshold] += threshold

    return mat


def compute_sparse_C(data, tau, mu2, verbose = False, epsilon = 1e-3):
    """
    Matrix LASSO Minimization by ADMM

    Parameters:
    -----------
    data:    array, shape[D, N]
             data matrix, N examples of dimension D
    tau:     positive real
             penalisation of errors for the sparse representation
    mu2:     postive real
             parameter of the augmented Lagrangian method
    verbose: boolean
             level of verbosity

    Returns:
    --------
    C:       array, shape[N, N]
             robust sparse representation of data
    """
    N = data.shape[1]

    # Crate C and lagrange multipliers
    C = np.zeros((N, N))
    lambda2 = np.zeros((N, N))
    Z = np.zeros((N, N))

    # Create same matrices to compare at previous stage
    C_previous = np.inf * np.ones((N, N))
    lambda2previous = np.inf * np.ones((N, N))
    Z_previous = np.inf * np.ones((N, N))

    # Invariant (left) part of the Z-update
    Z_1 = np.linalg.inv(tau * np.dot(data.T, data) + mu2 * np.eye(N))

    while (np.linalg.norm(C-C_previous) / (epsilon + np.linalg.norm(C)) \
    >epsilon \
    or np.linalg.norm(Z-Z_previous) / (epsilon + np.linalg.norm(Z))>epsilon):

        # Update previous matrices
        C_previous = np.array(C)
        Z_previous = np.array(Z)
        lambda2previous = np.array(lambda2)

        # Right part of the Z-update
        Z_2 = tau * np.dot(data.transpose(),data) + mu2*(C-lambda2/mu2)
        # Update Z
        Z = np.dot(Z_1, Z_2)

        # Update C
        C = soft_thresholding(Z + lambda2 / mu2, 1 / mu2)
        np.fill_diagonal(C, 0)

        # Update Lambda2
        lambda2 = lambda2 + mu2 * (Z - C)

        if verbose:
            # Show current error data - data.dot(C)
            print("Current error: {:.5e}"\
            .format(np.linalg.norm(data - data.dot(C))))

    return C


def SSC(data, n, tau, mu2, verbose = False):
    """
    Sparse Subspace algorithm

    Inspired from : algorithm 8.5
    Vidal, René, Yi Ma, and S. Shankar Sastry. "Principal Component Analysis." Generalized Principal Component Analysis. Springer New York, 2016

    Parameters:
    -----------
    data:    array, shape[D, N]
             data matrix, N examples of dimension D
    n:       positive integer
             number of subspaces
    tau:     positive real
             penalisation of errors for the sparse representation
    mu2:     postive real
             parameter of the augmented Lagrangian method
    verbose: boolean
             level of verbosity

    Returns:
    --------
    C:       array, shape[N, N]
             robust sparse representation of data
    groups:  array-like, shape (N,)
             labels (between 0 and n-1) of each point
    """
    # Compute robust sparse representation of data
    C = compute_sparse_C(data, tau, mu2, verbose = verbose)

    # Compute affinity matrix
    W = np.abs(C) + np.abs(C.transpose())

    # Compute D and L associated to graph
    D = np.diag(W.sum(axis = 1))
    L = D - W

    if verbose:
        print (D)

    # Normalized spectral clustering
    D_root = np.linalg.inv(scipy.linalg.sqrtm(D))
    _, evectors = np.linalg.eigh(np.dot(np.dot(D_root, L), D_root))
    kmeans = KMeans(n_clusters = n, init = 'random').fit(evectors[:, :n])

    return C, kmeans.labels_
