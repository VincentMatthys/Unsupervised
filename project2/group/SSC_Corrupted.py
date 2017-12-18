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


def compute_sparse_C(data,mu1,mu2):
    global epsilon
    C = np.zeros((data.shape[1],data.shape[1]))
    lambda2 = np.zeros((data.shape[1],data.shape[1]))
    Z= np.ones((data.shape[1],data.shape[1]))*float('inf')
    lambda1 = np.zeros(data.shape)

    Z_1 = np.linalg.inv(mu1*np.dot(data.transpose(),data)+ mu2*np.eye(data.shape[1]))
    itern = 0
    while (np.linalg.norm((Z-C))>epsilon)and(itern<10000):
        itern += 1

        Z_2 = mu1*np.dot(data.transpose(),data+lambda1/mu1) + mu2*(C-lambda2/mu2)
        Z = np.dot(Z_1,Z_2)

        C = S_Shrinkage(Z+lambda2/mu2,1/mu2)
        np.fill_diagonal(C,0)

        lambda1 = lambda1 + mu1*(data-np.dot(data,Z))

        lambda2 = lambda2 + mu2*(Z-C)

        print(np.linalg.norm((Z-C)))


    return C


def SSC(data,n,mu1,mu2):
    C = compute_sparse_C(data,mu1,mu2)

    W = abs(C) +abs(C.transpose())

    D = np.diag(W.sum(axis = 1))
    L = D - W


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
# mu2 = min(mu_list)
#
# C_m, eval_label = SSC(data,2,0.00001,20)
# true_label= true_label-1
# true_label= true_label.reshape(-1)
# error= evaluate_error(eval_label,true_label)
