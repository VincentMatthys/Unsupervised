from scipy.optimize import linear_sum_assignment


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
