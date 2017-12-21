from scipy.optimize import linear_sum_assignment
import numpy as np

def minWeightBipartiteMatching_2(clusteringA, clusteringB):
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

    if  nA != nB:
        raise ValueError("Lengths of clustering do no match")

    nC = max(len(idsA), len(idsB))
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

def evaluate_error(predictions, ground_truth, verbose = False):
    """
    Evaluate error for all possible permuatations of n letters
    -----------------------------------------------------------------------------------
    Parameters :
    ------------
    predictions : list of size K1
                  labels predicted from a clustering algorithm
    ground_truth : list of size K2
                   true labels

    Returns :
    ---------
    error : integer


    Sources :
    # 4.3.1
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.2605&rep=rep1&type=pdf
    # Other possible source
    https://www.r-bloggers.com/matching-clustering-solutions-using-the-hungarian-method/
    ------------------------------------------------------------------------------------
    """
    ## WARNING : only works if labels in range(d1), d1 positive integer
    # Construct the Hungarian matrix and get the mapping dictionnary
    dic = minWeightBipartiteMatching_2(predictions, ground_truth)

    # Translate predictions in term of ground_truth labels
    mapped_pred = predictions.copy()
    for k,i in enumerate(dic):
        mapped_pred[predictions == k] = i
    # Compute error
    error = (mapped_pred != ground_truth).sum() / len(ground_truth)

    # To do : plot confusion matrix
    if verbose:
        print ("Erreur : {}".format(error))
    return error
