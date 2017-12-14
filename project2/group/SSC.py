# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:44:23 2017

@author: Pirashanth
"""

import numpy as np
import scipy
from sklearn.cluster import KMeans
from scipy.io import loadmat
epsilon = 1


def S_Shrinkage(Matrix,tau):
    
    sign = lambda x : -1 if (x<0) else 1
    shrink = lambda m: sign(m)*max(abs(m)-tau,0)
    shrink_vect = np.vectorize(shrink)
    
    return shrink_vect(Matrix)


def compute_sparse_C(data,tau,mu2):
    global epsilon
    C = np.zeros((data.shape[1],data.shape[1]))
    lambda2 = np.zeros((data.shape[1],data.shape[1]))
    C_previous = np.ones((data.shape[1],data.shape[1]))*float('inf')
    lambda2previous = np.ones((data.shape[1],data.shape[1]))*float('inf')
    Z_previous= np.ones((data.shape[1],data.shape[1]))*float('inf')
    Z = np.zeros((data.shape[1],data.shape[1]))
    
    while (np.linalg.norm(C-C_previous)>epsilon or np.linalg.norm(Z-Z_previous)>epsilon or np.linalg.norm(lambda2-lambda2previous)>epsilon):
        
        C_previous = np.array(C)
        Z_previous = np.array(Z)
        lambda2previous = np.array(lambda2)
        
        
        Z_1 = np.linalg.inv(tau*np.dot(data.transpose(),data)+ mu2*np.eye(data.shape[1]))
        Z_2 = tau*np.dot(data.transpose(),data) + mu2*(C-lambda2/mu2)
        Z = np.dot(Z_1,Z_2)
        C = S_Shrinkage(Z+lambda2/mu2,1/mu2)
        C = C - np.diag(C)
        lambda2 = lambda2 + mu2*(Z-C)
        print(np.linalg.norm(C-C_previous),np.linalg.norm(Z-Z_previous),np.linalg.norm(lambda2-lambda2previous))

    
    return C
        

def SSC(data,n,tau,mu2):
    C = compute_sparse_C(data,tau,mu2)
    
    W = abs(C) +abs(C.transpose())
    
    D = np.diag(np.sum(W,axis=0))
    
    L = D - W
    
    D_1_2= np.linalg.inv(scipy.linalg.sqrtm(D))
    
    transform_L= np.dot(np.dot(D_1_2,L),D_1_2)
    w,v = np.linalg.eig(transform_L)
    
    eigv = v[-n:,:]
    row_sums = eigv.sum(axis=1)
    normalize_eigv = eigv / row_sums[:, np.newaxis]
    
    km = KMeans(n_clusters=n)
    km.fit(normalize_eigv.transpose())
    return C,km.predict(normalize_eigv.transpose())
     

    
mat = loadmat('ExtendedYaleB.mat')
data = mat['EYALEB_DATA']
true_label = mat['EYALEB_LABEL']
C_m, eval_label = SSC(data,16,0.5,0.1)
