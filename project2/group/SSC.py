# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:44:23 2017

@author: Pirashanth
"""

import numpy as np
import scipy
from sklearn.cluster import KMeans
from scipy.io import loadmat
from error_evaluation import evaluate_error

epsilon = 1e-3


def S_Shrinkage(Matrix,tau):

    sign = lambda x : -1 if (x<0) else 1
    shrink = lambda m: sign(m)*max(abs(m)-tau,0)
    shrink_vect = np.vectorize(shrink)

    return shrink_vect(Matrix)

def soft_thresholding(mat, epsilon):
    """
    Soft thresholding operator
    Proximal operator of the L1 norm
    :param mat: matrix to threshold
    :param epsilon: value of threshold
    :return mat: thresholded matrix
    """

    mat[np.abs(mat) < epsilon] = 0
    mat[mat > epsilon] -= epsilon
    mat[mat < -epsilon] += epsilon

    return mat


def compute_sparse_C(data,tau,mu2, verbose = False):
    global epsilon
    C = np.zeros((data.shape[1],data.shape[1]))
    lambda2 = np.zeros((data.shape[1],data.shape[1]))
    C_previous = np.inf * np.ones((data.shape[1],data.shape[1]))
    lambda2previous = np.inf * np.ones((data.shape[1],data.shape[1]))
    Z_previous = np.inf * np.ones((data.shape[1],data.shape[1]))
    Z = np.zeros((data.shape[1],data.shape[1]))


    Z_1 = np.linalg.inv(tau*np.dot(data.transpose(),data)+ mu2*np.eye(data.shape[1]))

    while (np.linalg.norm(C-C_previous) / (epsilon + np.linalg.norm(C)) >epsilon or np.linalg.norm(Z-Z_previous) / (epsilon + np.linalg.norm(Z))>epsilon):

        C_previous = np.array(C)
        Z_previous = np.array(Z)
        lambda2previous = np.array(lambda2)


        Z_2 = tau * np.dot(data.transpose(),data) + mu2*(C-lambda2/mu2)
        Z = np.dot(Z_1,Z_2)

        # C = S_Shrinkage(Z+lambda2/mu2,1/mu2)
        ## %lprun -f  compute_sparse_C T = SSC(motion_data[key]["data"], motion_data[key]["s"].max() + 1, 1e7, 1e3) => 115s 99% S_Shrinkage
        C = soft_thresholding(Z+lambda2/mu2,1/mu2)
        ## %lprun -f  compute_sparse_C T = SSC(motion_data[key]["data"], motion_data[key]["s"].max() + 1, 1e7, 1e3) => 7.68323 s 25% S_Shrinkage
        np.fill_diagonal(C,0)

        lambda2 = lambda2 + mu2*(Z-C)

        if verbose:
            print(np.linalg.norm(C-C_previous),np.linalg.norm(Z-Z_previous),np.linalg.norm(lambda2-lambda2previous))


    return C


def SSC(data,n,tau,mu2):
    C = compute_sparse_C(data,tau,mu2)

    W = np.abs(C) + np.abs(C.transpose())

    D = np.diag(W.sum(axis = 1))
    L = D - W


    print (D)
    D_1_2= np.linalg.inv(scipy.linalg.sqrtm(D))

    transform_L= np.dot(np.dot(D_1_2,L),D_1_2)
    evalues, evectors = np.linalg.eigh(transform_L)


    Y = evectors[:, :n].T

    kmeans = KMeans(n_clusters = n, init = 'random').fit(Y.T)

    return C,kmeans.labels_


# mat = loadmat('ExtendedYaleB.mat')
# data = mat['EYALEB_DATA']
# true_label = mat['EYALEB_LABEL']
# data = np.array(data, dtype = np.int64)
# true_label = true_label[:,:128]
#
# data=data[:,0:128]
# N=data.shape[1]
# D=data.shape[0]
#
# #compute mu_min
#
# l=np.zeros((N,N))
# mu_list = np.zeros((N,1))
# for i in range(N):
#     for j in range(N) :
#         l[i,j] = np.dot(data[:,i].transpose(), data[:,j])
# np.fill_diagonal(l,0)
#
# for i in range(N):
#     mu_list[i,0] = max(l[i,:])
#
# maxmin = min(mu_list)
#
# mu2 = 20
# tau = 1e-5
#
#
# C_m, eval_label = SSC(data,2,tau,mu2)
# true_label= true_label-1
# true_label= true_label.reshape(-1)
# error= evaluate_error(eval_label,true_label)
