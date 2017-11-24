import numpy as np

# To find pgm files
import glob

# To read and show pgm files
from skimage.io import imread, imshow

import matplotlib.pyplot as plt

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

def svalue_thresholding(mat, epsilon):
    """
    Singular value thresholding
    Proximal operator of the nuclear norm
    :param mat: matrix of svalue to threshold
    :param epsilon: value of threshold
    :return mat: thresholded matrix
    """
    u, s, v = np.linalg.svd(mat, full_matrices = False, compute_uv = 1)
    soft_thresholding(s, epsilon)
    return u.dot(np.diag(s).dot(v))

def lrmc(X, W, tau, beta, tol = 100, A = 0):
    """
    Finds a low-rank matrix A whose entries in W coincide
    with those of X by using the SVT algorithm.
    :param X: DxN data matrix
    :param W: DxN binary matrix denoting known (1) or missing (0) entries
    :param tau: Parameter of the optimization problem
    :param beta: Step size of the dual gradient ascent step
    :param tol: tolerance for convergence boost speed
    :return A: Low-rank completion of the mtrix X
    """
    if (A == 0).all():
        Z = 0
    else :
        Z = beta * (X * W - A * W)
    Z_stop = tol
    while (np.linalg.norm(Z - Z_stop) >= tol):
        print (np.linalg.norm(Z - Z_stop))
        Z_stop = Z
        A = svalue_thresholding(Z * W, tau)
        Z = Z + beta * (X * W - A * W)
    return A


IMG_DIR = "images/"
INDIVIDUALS = ["yaleB01", "yaleB02", "yaleB03"]

if __name__ == '__main__':
	faces = {}
    for k in INDIVIDUALS:
        faces[k] = {}
        faces[k]['files'] = glob.glob(IMG_DIR + k + "/" + k + "_P00A" + "*.pgm")
        faces[k]["data"] = np.concatenate([imread(f).reshape(-1, 1)\
                                           for f in faces[k]['files']], axis = 1)
        # Draw uniformly at random
        for i in np.arange(10) * 10 :
            faces[k][i] = np.random.choice([0,1],
                                           size = faces[k]["data"].shape,
                                           p = [i/100, 1 - i/100])

    D = faces['yaleB01']
    MSE = {}
    RES = {}
    n_cols = 64
    # tau_range = [50000, 100000, 200000, 400000, 600000]
    tau_range = [1000, 10000, 50000, 100000, 2000000, 300000, 400000, 500000,     600000, 700000, 1000000]
    for M in np.arange(1, 10) * 10.:
        print ("--------------- % of missing entries : {}         ---------------".format(M/100))
        MSE[M] = {}
        RES[M] = {}
        for t in range(len(tau_range)):
            print ("Power for entrepreneurs : {}".format(tau_range[t]))
            RES[M][tau_range[t]] = lrmc(D["data"][:, :n_cols], D[M][:, :n_cols],
                             tau = tau_range[t],
                             beta = min(2, 1/(1-M/100)),
                             tol = 1,
                             A = RES[M][tau_range[t - 1]] if t > 0 else     np.zeros((1,1)))
            MSE[M][tau_range[t]] = ((D["data"][:, :n_cols] - RES[M][tau_range[t]]) **2).mean(axis = 0)

    np.save("RES_1.npy", RES)
    np.save("MSE_1.npy", RES)
