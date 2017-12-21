import numpy as np

def ksubspaces(data, n , d, replicates, verbose = False, epsilon = 1e-3):
    """
    K-subspaces algorithm

    Inspired from : algorithm 6.1
    Vidal, René, Yi Ma, and S. Shankar Sastry. "Principal Component Analysis." Generalized Principal Component Analysis. Springer New York, 2016

    Parameters:
    ------------
    data:           array, shape [D, N]
                    data matrix, N examples of dimension D
    n:              postive integer
                    number of subspaces
    d:              list, array-like, shape (n,)
                    dimension of subpsaces
    replicates:     number of restarts
    verbose:        boolean
                    level of verbosity
    epsilon:		positive real
                    Parameter for stop condition

    Returns:
    --------
    global_groups:  list:
                      - array-like, shape (N,)
                        segmentation of the data in n groups,
                        labels (between 0 and n-1) for each point
                      - array, shape [n, N]
                        segmentation of of the data
    global_objects: list:
                      - list of U_i, subspace bases
                      - array of mus, subspace barycenters
                      - low-dimensional representations of the data
                    subspaces parameters and low-dimensional representations

    """
    D = data.shape[0]
    N = data.shape[1]

    # Keep trace of every replicate
    trace = {r: {} for r in range(replicates)}
    # Iterating over repliactes
    for r in range(replicates):

        ## randomly selecting mu as initial points in data
        mu = data[:, np.random.choice(N, n, replace = False)]

        ## randomly selecting U
        U = [np.random.randn(D, d_u) for d_u in d]
        U_norm = [u / np.linalg.norm(u, axis = 0, keepdims = True) for u in U]

        ## Previous state of mu and U (subspaces parameters)
        U_prev = [np.inf * np.ones((D, d_u)) for d_u in d]
        mu_prev = np.inf * np.ones(mu.shape)

        while (np.linalg.norm(mu-mu_prev) / (epsilon + np.linalg.norm(mu)) >\
        epsilon\
        or np.array([np.linalg.norm(U[k] - U_prev[k]) for k in range(n)]).sum()\
        / (epsilon + np.array([np.linalg.norm(U[k]) for k in range(n)]).sum())\
        > epsilon):
            ### Update previous parameters
            U_prev = U
            mu_prev = mu
            w = np.zeros((n,N))

            ### Segmentation
            distance = np.zeros((n, N))
            ### Invariant Left part of the argmin search
            left = np.eye(D) - np.array([u.dot(u.T) for u in U_norm])
            #### Search for each point the closest subspace
            for j in range(N):
                distance[:, j] = np.linalg.norm(np.matmul(left, (data[:,j]\
                .reshape(-1, 1) - mu).T.reshape(n, D, 1)), axis = 1).reshape(-1)
                w[distance[:, j].argmin(), j] = 1

            ### Estimation
            ### mus-estimation
            mu = w.dot(data.T).T / np.maximum(w.sum(axis = 1), 1)
            ### Us-estimation
            #### For each cluster, pick the top d[i] evectors
            for i in range(n):
                Q = data[:, w[i, :].astype(bool)].T.reshape(-1, D, 1)
                QQ = np.matmul(Q, np.transpose(Q, axes = (0, 2, 1))).sum(axis = 0)
                _, v = np.linalg.eigh(QQ)
                U[i] = v[:, -int(d[i]):]

            if verbose:
                # Show current error
                print ("Current error for {} replicate : {:.5e}".format(r, np.linalg.norm(w*distance)))

        # Update trace for this replicate
        trace[r]["w"] = w
        trace[r]["U"] = U
        trace[r]["mu"] = mu
        trace[r]["error"] = np.linalg.norm(w*distance)


    # Select best replicate
    best = min(trace, key = lambda k: trace[k]["error"])

    if verbose:
        print("Best replicate {} with error {}"\
        .format(best, trace[best]["error"]))

    pos = trace[best]["w"].argmax(0)
    y = [trace[best]["U"][pos[j]].T.dot(data[:,j] - mu[:, pos[j]]) for j in range(N)]


    return (trace[best]["w"].argmax(0), trace[best]["w"]),\
            [trace[best]["U"], trace[best]["mu"], y]
