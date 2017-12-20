import numpy as np

epsilon = 1e-3


def ksubspaces2(data, n , d, replicates):
    """
    K-subspaces algorithm

    Parameters:
    ------------
    data:           array, shape [D, N]
                    data matrix, N examples of dimension D
    n:              postive integer
                    number of subspaces
    d:              list, array-like, shape (n,)
                    dimension of subpsaces
    replicates:     number of restarts

    Returns:
    --------
    global_groups:
    global_objects:

    """
    D = data.shape[0]
    N = data.shape[1]

    err = []
    for r in range(replicates):

        mu = data[:, np.random.choice(N, n, replace = False)]
        mu_prev = np.ones(mu.shape)


        ### randomly selecting U
        U = [np.random.randn(D, d_u) for d_u in d]
        U_norm = [u / np.linalg.norm(u, axis = 1, keepdims = True) for u in U]

        U_prev = np.ones((n,D,int(d[0])))
        y = np.zeros((D,N))

#         print('mu Error : ' , np.linalg.norm(mu-mu_prev) , 'U error', np.linalg.norm(U-U_prev))

        while (np.linalg.norm(mu-mu_prev) / (epsilon + np.linalg.norm(mu)) > epsilon\
               or np.array([np.linalg.norm(U[k] - U_prev[k]) for k in range(n)]).sum()\
               / (epsilon + np.array([np.linalg.norm(U[k]) for k in range(n)]).sum()) > epsilon):
            U_prev = U
            mu_prev = mu
            w = np.zeros((n,N))

            ##used for multiple restart
            distance = np.zeros((n, N))
            left = np.eye(D) - np.array([u.dot(u.T) for u in U_norm])
            ## Segmentation
            for j in range(N):
                distance[:, j] = np.linalg.norm(np.matmul(left, (data[:,j]\
                .reshape(-1, 1) - mu).T.reshape(n, D, 1)), axis = 1).reshape(-1)
                w[distance[:, j].argmin(), j] = 1

            U = [np.zeros((D, d_u)) for d_u in d]
            # Estimation
            mu = w.dot(data.T).T / np.maximum(w.sum(axis = 1), 1)
            for i in range(n):
                Q = data[:, w[i, :].astype(bool)].T.reshape(-1, D, 1)
                QQ = np.matmul(Q, np.transpose(Q, axes = (0, 2, 1))).sum(axis = 0)

                e, v = np.linalg.eigh(QQ)
                U[i] = v[:,-int(d[i]):]

            print ("Current error for {} replicate : {}".format(r, np.linalg.norm(w*distance)))


        pos = w.argmax(0)
        y = [U[pos[j]].T.dot(data[:,j] - mu[:, pos[j]]) for j in range(N)]

        error_cur = np.linalg.norm(w * distance)
        err.append(error_cur)

        if (error_cur == min(np.array(err))):
            w_opt = w
            mu_opt = mu
            U_opt = U
            y_opt = y

    print (err)

    return w_opt, mu_opt, U_opt, y_opt
