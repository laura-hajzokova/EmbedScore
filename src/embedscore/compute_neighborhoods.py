import numpy as np
from typing import Union
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold._t_sne import _joint_probabilities

def get_neighbors(D, k=None):
    '''Get indices of the k nearest neighbours of each point
    
    Parameters
    ----------
    D    - distance matrix of the shape (N,N)
    k    - the numer of nearest neighbors

    Return
    ----------
    idcs - indices of f the neighbours for each point, matrix of the shape (N,k) 
    '''
    R = np.argsort(D, axis=1)
    return R[:,1:k+1] if k is not None else R

def extract_neighbors_dist(D: np.ndarray,
                           R: np.ndarray, 
                           idcs_subset: Union[list, np.array]=None) -> np.ndarray:
    """
    Extract the distances to the nearest neighbors for each point or the subset of points.

    Parameters
    ----------
    D       - np.ndarray of shape (N, N)
    R       - np.ndarray of shape (N, k), integer array where each row i
              contains the k column indices of the nearest neighbours for each point from D.
    idcs    - indices of the subsetted points

    Returns
    -------
    result  : np.ndarray of shape (N, K); distances to the K nearest neighbors.
    """
    assert D.ndim == 2 and D.shape[0] == D.shape[1], "D must be square (N, N)"
    assert R.ndim == 2 and R.shape[0] == D.shape[0], "indices must have shape (N, K) where N matches D"

    N = D.shape[0]
    K = R.shape[1]

    # If subsetting
    if idcs_subset is not None:
        distances = D[np.array(idcs_subset).reshape(-1,1),R[idcs_subset, 0:K]]  # Exclude self (first neighbor)
    else:
        distances = D[np.arange(D.shape[0]).reshape(-1, 1), R[:, 0:K]]

    return distances

def extract_neighbors_emb(R: np.ndarray,
                          idcs_subset: Union[list, np.ndarray]) -> np.ndarray:
    """
    Extract the nearest neighbors for the subset of points.

    Parameters
    ----------
    R       - np.ndarray of shape (N, k), integer array where each row i
              contains the k column indices of the nearest neighbours for each point from D.
    idcs    - indices of the subsetted points

    Returns
    -------
    result  : np.ndarray of shape (N, 2); coordinates of the HD neighbors of the points from the subset.
    """
    
    K = R.shape[1]

    if len(idcs_subset) == 1:
        neighbors = R[idcs_subset, 1:K]
    else:
        neighbors = R[idcs_subset, 1:K].flatten()
    neighbors = np.unique(neighbors)
    
    return neighbors

def delaunay_distance_matrix(data, adj: bool = False):
    tri = Delaunay(data)
    indptr, indices = tri.vertex_neighbor_vertices
    n = data.shape[0]
    ones_n = np.ones(len(indices), dtype=int)
    # Create the sparse adjacency matrix
    adjacency_matrix = csr_matrix((ones_n, indices, indptr), shape=(n, n))
    distance_matrix = floyd_warshall(csgraph=adjacency_matrix, directed=False)

    if adj:
        return distance_matrix, adjacency_matrix
    else:
        # Compute the distance matrix
        return distance_matrix
    
def distance_to_neighbors(D: np.ndarray, R: np.ndarray, K: int=None):
    N = D.shape[0]
    mask = np.zeros((N, N), dtype=bool)
    if K is not None:
        R = R[:,1:K+1]
    mask[np.arange(N)[:, None], R] = True
    D_new = np.where(mask, D, 0)
    return D_new

def gaussian_distribution(D: np.ndarray, perplexity: int=30, tol: float=1e-5):
    '''P is a gaussian distribution

    Parameters
    ----------
    D       - np.ndarray of shape (N, N), distance matrix

    Returns
    -------
    P       - np.ndarray of shape (N, N), the matrix of pairwise similarities in the high-dimensional space
    '''

    P = _joint_probabilities(D, perplexity, tol)
    return squareform(P)

def student_t_distribution(D: np.ndarray):
    '''Q is a heavy-tailed distribution: Student's t-distribution

    Parameters
    ----------
    D       - np.ndarray of shape (N, N) or a condensed matrix of shape (N*(N-1)/2,), distance matrix

    Returns
    -------
    Q       - np.ndarray of shape (N, N), the matrix of pairwise similarities in the low-dimensional space
    '''

    if D.ndim == 2:
        assert D.shape[0] == D.shape[1], "D must be square (N, N)"
        D = squareform(D)

    machine_epsilon = np.finfo(np.double).eps
    D /= 1
    D += 1.0
    D **= (1 + 1.0) / -2.0
    Q = np.maximum(D / (2.0 * np.sum(D)), machine_epsilon)

    return squareform(Q)

