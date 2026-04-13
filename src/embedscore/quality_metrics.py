from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy.linalg as LA   
from .compute_neighborhoods import get_neighbors, extract_neighbors_dist

rng = np.random.default_rng(42)

def stress(D_hd: np.ndarray, D_ld: np.ndarray) -> tuple:
    '''
    Get the stress metric

    Parameters
    ----------
    D_hd - distance matrix of the original data
    D_ld - distance matrix of the embedding

    Return
    ------
    links - quality of links between points in the original data and the embedding
    nodes - quality of nodes in the original data and the embedding
    map - quality of the map (stress metric)
    '''

    D_hd = np.array(D_hd)
    D_ld = np.array(D_ld)
    
    assert D_hd.shape == D_ld.shape

    # normalisation factor
    links = (D_hd - D_ld) / np.sqrt(np.sum(D_hd**2))
    nodes = np.sum(links**2, axis=1)
    map = np.sqrt(np.sum(nodes))

    return links, nodes, map

def precision_maps(D_hd, D_ld, K=None):
    '''
    Get the precision maps for the embedding

    Parameters
    ----------
    D_hd - distance matrix of the original data
    D_ld - distance matrix of the embedding
    k - number of neighbors to consider

    Return
    ------
    links - quality of links between points in the original data and the embedding
    nodes - quality of nodes in the original data and the embedding
    map - quality of the map (precision metric)
    '''

    D_hd = np.array(D_hd)
    D_ld = np.array(D_ld)
    
    assert D_hd.shape == D_ld.shape

    N = len(D_hd)
    if K is None:
        K = N * 0.02

    R_hd = np.argsort(D_hd, axis=1)
    R_ld = np.argsort(D_ld, axis=1)

    links = np.zeros_like(D_hd)

    for i in range(N):
        norm_hd = LA.norm(D_hd[i, R_hd[i, :K]])
        norm_ld = LA.norm(D_ld[i, R_hd[i, :K]])
        if norm_hd > 0 and norm_ld > 0:
            links[i, R_hd[i, :K]] = (D_hd[i, R_hd[i, :K]]/norm_hd) - (D_ld[i, R_hd[i, :K]]/norm_ld)

    nodes = LA.norm(links, axis=1)
    map = None

    return links, nodes, map

def trustworthiness(D_hd, D_ld, idcs=None, K=None):
    '''
    Get the trustworthiness metric

    Parameters
    ----------
    D_hd - distance matrix of the original data
    D_ld - distance matrix of the embedding
    k - number of neighbors to consider

    Return
    ------
    links - quality of links between points in the original data and the embedding
    nodes - quality of nodes in the original data and the embedding
    map - quality of the map (trustworthiness metric)
    '''

    D_hd = np.array(D_hd)
    D_ld = np.array(D_ld)
    
    assert D_hd.shape == D_ld.shape

    N = len(D_hd)

    R_hd = np.argsort(np.argsort(D_hd, axis=1), axis=1)
    R_ld = np.argsort(D_ld, axis=1)

    # normalisation factor
    if K < N/2:
        norm1 = 2 / (2*N - 3*K - 1)
        norm2 = 1 / K
    else:
        norm1 = 2 / (N-K-1)
        norm2 = 1 / (N-K)

    if idcs is not None:
        R_hd = R_hd[idcs,:]
        links = np.maximum(R_hd - K, 0) * norm1
        idx0 = 0
        for i in idcs:
            for j in range(K, N):
                idx = R_ld[i, j]
                links[idx0, idx] = 0
            idx0 += 1
    else:
        links = np.maximum(R_hd - K, 0) * norm1
        for i in range(N):
            for j in range(K, N):
                idx = R_ld[i, j]
                links[i, idx] = 0
                
    nodes = np.sum(links, axis=1) * norm2
    map = np.sum(nodes)/N

    return links, nodes, map

def continuity(D_hd, D_ld, idcs=None, K=None):
    '''
    Get the continuity metric

    Parameters
    ----------
    D_hd - distance matrix of the original data
    D_ld - distance matrix of the embedding
    k - number of neighbors to consider

    Return
    ------
    links - quality of links between points in the original data and the embedding
    nodes - quality of nodes in the original data and the embedding
    map - quality of the map (continuity metric)
    '''

    D_hd = np.array(D_hd)
    D_ld = np.array(D_ld)
    
    assert D_hd.shape == D_ld.shape

    N = len(D_hd)

    R_hd = np.argsort(D_hd, axis=1)
    R_ld = np.argsort(np.argsort(D_ld, axis=1), axis=1)

    # normalisation factor
    if K < N/2:
        norm1 = 2 / (2*N - 3*K - 1)
        norm2 = 1 / K
    else:
        norm1 = 2 / (N-K-1)
        norm2 = 1 / (N-K)

    if idcs is not None:
        R_ld = R_ld[idcs,:]
        links = np.maximum(R_ld - K, 0) * norm1
        idx0 = 0
        for i in idcs:
            for j in range(K, N):
                idx = R_hd[i, j]
                links[idx0, idx] = 0
            idx0 += 1
    else:
        links = np.maximum(R_ld - K, 0) * norm1
        for i in range(N):
            for j in range(K, N):
                idx = R_hd[i, j]
                links[i, idx] = 0
                
    nodes = np.sum(links, axis=1) * norm2
    map = np.sum(nodes)/N

    return links, nodes, map



def link_stress(D_hd: np.ndarray, D_ld: np.ndarray, method: str = 'classic', norm: Union[np.ndarray, float] = None) -> np.ndarray:
    '''Computes links based on the Kruskall's stress function

    Parameters
        ----------
        D_hd    - distance matrix of the original data of shape (N, N)
        D_ld    - distance matrix of the embedding of shape (N, M)
        method  - method to use for the links ('classic','nonmetric', 'isomap', 'sammon')
        norm    - normalisation factor for the distances in HD and LD space of shape (N,N) or scalar

        Return
        ------
        links   - quality of links between points in the original data and the embeddings
    '''

    if norm is None:
        if method == 'classic':
            norm = np.sqrt(np.sum(D_hd**2))                # Classic MDS
            links = (D_hd - D_ld) / norm
        elif method == 'nonmetric':                        # Nonmetric MDS - UNFINISHED!!
            'shephard\'s stress'
        elif method == 'sammon':                           # Sammon mapping
            norm = D_hd * np.sum(D_hd)
            links = (D_hd - D_ld) / norm
        elif method == 'isomap':                           # Isomap - UNFINISHED!!
            'geodesic dist.'
        else: 
            print('Invalid method')
    elif norm.shape == D_hd.shape or type(norm) == float:  # allows for different normalisation
        links = (D_hd - D_ld) / norm
    else:
        print("Normalising factor has to be of shape (N,N) or a scalar ")

    return links

def link_precision_maps(D_hd: np.ndarray, D_ld: np.ndarray, r_hd: np.ndarray = None, norm: Union[np.ndarray, float] = None, K: int = 100) -> np.ndarray:
    '''Computes links based on the precision maps criterion from [1]

    [1] Schreck, T., von Landesberger, T., and Bremm, S. (2010). Techniques for precision-based visual analysis of projected data.

    Parameters
        ----------
        D_hd    - distance matrix of the original data of shape (N, N)
        D_ld    - distance matrix of the embedding of shape (N, N)
        r_hd    - numpy array (N,N), rank matrix of the original data
        norm    - normalisation factor for the distances in HD and LD space of shape (2,N,N) 
        K       - neighborhood size

        Return
        ------
        links   - quality of links between points in the original data and the embeddings
    '''
    try:
        assert D_hd.shape == D_ld.shape, "Distance matrices must have equal shape"
    except AssertionError as e:
        print(f"Assertion failed: {e}")

    N = len(D_hd)

    if norm is None:
        R = np.argsort(D_hd, axis=1)[:, 1:K+1] if r_hd is None else r_hd[:, 1:K+1] # indices of the K nearest neighbors in HD space
        mask = np.zeros((N, N), dtype=bool)
        mask[np.arange(N)[:, None], R] = True
        D_hd = np.where(mask, D_hd, 0)
        D_ld = np.where(mask, D_ld, 0)
        norm_hd = np.sqrt(np.sum(D_hd**2, axis=1)).reshape(-1, 1)
        norm_ld = np.sqrt(np.sum(D_ld**2, axis=1)).reshape(-1, 1)
        links = (D_hd/norm_hd) - (D_ld/norm_ld)
    elif norm.shape == (2, D_hd.shape[0], D_hd.shape[0]):
        links = (D_hd/norm_hd) - (D_ld/norm_ld)
    elif norm.shape == D_hd.shape or type(norm) == float:  # allows for different normalisation
        links = (D_hd - D_ld) / norm
    else:
        print("Normalising factor has to be of shape (N,N) or a scalar ")

    return links

def link_projection_error(D_hd: np.ndarray, D_ld: np.ndarray, norm: Union[np.ndarray, float] = None) -> np.ndarray:
    '''Computes links based on the projection error criterion from [2]

    [2]  Martins, R. M., Coimbra, D. B., Minghim, R., and Telea, A. (2014). Visual analysis of dimensionality reduction quality for parameterized projections. 

    Parameters
        ----------
        D_hd    - distance matrix of the original data of shape (N, N)
        D_ld    - distance matrix of the embedding of shape (N, N)
        norm    - normalisation factor for the distances in HD and LD space of shape (2,) 

        Return
        ------
        links   - quality of links between points in the original data and the embeddings
    '''
    try:
        assert D_hd.shape == D_ld.shape, "Distance matrices must have equal shape"
    except AssertionError as e:
        print(f"Assertion failed: {e}")

    if norm is None:
            norm_hd = np.max(D_hd)
            norm_ld = np.max(D_ld) 
            links = (D_hd/norm_hd) - (D_ld/norm_ld)
    elif norm.shape == (2,):                         
        links = (D_hd/norm[0] - D_ld/norm[1])
    else:
        print("Normalising factor has to be of shape (2,)")

    return links

def link_trustworthiness(D_hd: np.ndarray, D_ld: np.ndarray, r_hd: np.ndarray = None, r_ld: np.ndarray = None, K: int = 100) -> np.ndarray:
    '''Computes links based on the trustworthiness criterion from [3]

    [3]  Venna, J. and Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: An experimental study.

    Parameters
        ----------
        D_hd    - numpy array (N,N), distance matrix of the original data
        D_ld    - numpy array (N,N), distance matrix of the embedding
        r_hd    - numpy array (N,N), rank matrix of the original data
        r_ld    - numpy array (N,N), rank matrix of the embedding
        K       - neighborhood size

        Return
        ------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings
    '''
    try:
        assert D_hd.shape == D_ld.shape, "Distance matrices must have equal shape"
    except AssertionError as e:
        print(f"Assertion failed: {e}")

    N = len(D_hd)

    if r_hd is None:
        R_hd = np.argsort(np.argsort(D_hd, axis=1), axis=1) # ranks of the HD neighbors
    else:
        R_hd = np.argsort(r_hd, axis=1)                     # ranks of the HD neighbors
    
    if r_ld is None:
        r_ld = np.argsort(D_ld, axis=1)                     # indices of the LD neighbors

    # normalisation factor
    if K < N/2:
        norm1 = 2 / (2*N - 3*K - 1)
        #norm2 = 1 / K
    else:
        norm1 = 2 / (N-K-1)
        #norm2 = 1 / (N-K)
      

    links = np.maximum(R_hd - K, 0) * norm1     # take the ranks of the points that are not in the `hd neighborhood`

    idx = r_ld[:,K+1:N]
    z = np.zeros((N,len(np.arange(K+1,N))))
    links[np.arange(N)[:, None], idx] = z
    
    return links

def link_continuity(D_hd: np.ndarray, D_ld: np.ndarray, r_hd: np.ndarray = None, r_ld: np.ndarray = None, K: int = 100) -> np.ndarray:
    '''Computes links based on the continuity criterion from [3]

    [3]  Venna, J. and Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: An experimental study.

    Parameters
        ----------
        D_hd    - numpy array (N,N), distance matrix of the original data
        D_ld    - numpy array (N,N), distance matrix of the embedding
        r_hd    - numpy array (N,N), rank matrix of the original data
        r_ld    - numpy array (N,N), rank matrix of the embedding
        K       - neighborhood size

        Return
        ------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings
    '''
    try:
        assert D_hd.shape == D_ld.shape, "Distance matrices must have equal shape"
    except AssertionError as e:
        print(f"Assertion failed: {e}")

    N = len(D_hd)

    if r_hd is None:
        r_hd = np.argsort(D_hd, axis=1)                     # indices of the HD neighbors
    
    if r_ld is None:
        R_ld = np.argsort(np.argsort(D_ld, axis=1), axis=1) # ranks of the LD neighbors
    else:
        R_ld = np.argsort(r_ld, axis=1)                     # ranks of the LD neighbors

    # normalisation factor
    if K < N/2:
        norm1 = 2 / (2*N - 3*K - 1)
        #norm2 = 1 / K
    else:
        norm1 = 2 / (N-K-1)
        #norm2 = 1 / (N-K)

    links = np.maximum(R_ld - K, 0) * norm1     # take the ranks of the points that are not in the `hd neighborhood`

    idx = r_hd[:,K+1:N]
    z = np.zeros((N,len(np.arange(K+1,N))))
    links[np.arange(N)[:, None], idx] = z
    
    return links

def link_mrre(D_hd: np.ndarray, D_ld: np.ndarray, r_hd: np.ndarray = None, r_ld: np.ndarray = None, K: int = 100, method = 'intrusions') -> np.ndarray:
    '''Computes links based on the MRRE (mean relative rank error) criterion from [4]

    [4]   Lee, J. A. and Verleysen, M. (2007). Nonlinear Dimensionality Reduction.

    Parameters
        ----------
        D_hd    - numpy array (N,N), distance matrix of the original data
        D_ld    - numpy array (N,N), distance matrix of the embedding
        r_hd    - numpy array (N,N), rank matrix of the original data
        r_ld    - numpy array (N,N), rank matrix of the embedding
        K       - neighborhood size
        method  - evaluation method, {'intrusions', 'extrusions'}

        Return
        ------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings
    '''
    try:
        assert D_hd.shape == D_ld.shape, "Distance matrices must have equal shape"
    except AssertionError as e:
        print(f"Assertion failed: {e}")

    if r_hd is not None:
        assert r_hd.shape == D_hd.shape, "r_hd must have the same shape as D_hd"
    else:
        r_hd = np.argsort(D_hd, axis=1) # indices of the HD neighbors
    if r_ld is not None:
        assert r_ld.shape == D_ld.shape, "r_ld must have the same shape as D_ld"
    else:        
        r_ld = np.argsort(D_ld, axis=1) # indices of the LD neighbors

    N = len(D_hd)

    R_hd = np.argsort(r_hd, axis=1) # ranks of the HD neighbors
    R_ld = np.argsort(r_ld, axis=1) # ranks of the LD neighbors

    # normalisation factor
    norm1 = K / abs(N - 2*K + 1)
    #norm2 = 1 / N

    if method == 'intrusions':
        with np.errstate(divide='ignore', invalid='ignore'):
            norm1 = norm1 * np.where(R_ld != 0, norm1 / R_ld, 0)  # 0 is the fallback value
        idx = r_ld[:,K+1:N]
    elif method == 'extrusions':
        with np.errstate(divide='ignore', invalid='ignore'):
            norm1 = norm1 * np.where(R_hd != 0, norm1 / R_hd, 0)  # 0 is the fallback value
        idx = r_hd[:,K+1:N]
      
    links =  np.abs(R_hd-R_ld) * norm1   # take the ranks of the points that are not in the `hd neighborhood`

    z = np.zeros((N,len(np.arange(K+1,N))))
    links[np.arange(N)[:, None], idx] = z
    
    return links



def nodes_stress(links: np.ndarray) -> np.ndarray:
    '''Computes nodes based on the stress criterion

    Parameters      
        ----------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings

        Return
        ------
        nodes   - numpy array (N,), quality of nodes in the original data and the embeddings
    '''

    return np.sum(links**2, axis=1)

def nodes_precision_maps(links: np.ndarray) -> np.ndarray:
    '''Computes nodes based on the precision maps criterion

    Parameters      
        ----------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings

        Return
        ------
        nodes   - numpy array (N,), quality of nodes in the original data and the embeddings
    '''

    return LA.norm(links, axis=1)

def nodes_projection_error(links: np.ndarray) -> np.ndarray:
    '''Computes nodes based on the projection error criterion

    Parameters      
        ----------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings

        Return
        ------
        nodes   - numpy array (N,), quality of nodes in the original data and the embeddings
    '''

    return LA.norm(links, 1, axis=1)

def nodes_rank_criteria(links: np.ndarray=None, r_hd: np.ndarray = None, r_ld: np.ndarray = None, method: str = 'trustworthiness', K: int = 100) -> np.ndarray:
    '''Computes nodes based on the trustworthiness and continuity criterion

    Parameters      
        ----------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings
        r_hd    - numpy array (N,N), rank matrix of the original data
        r_ld    - numpy array (N,N), rank matrix of the embedding
        method  - evaluation method, {'trustworthiness', 'continuity', 'mrre', 'lcmc', 'jaccard'}

        Return
        ------
        nodes   - numpy array (N,), quality of nodes in the original data and the embeddings
    '''

    if method in ['trustworthiness', 'continuity', 'mrre']:
        assert links is not None, "Links must be provided for trustworthiness, continuity and mrre"
        return np.sum(links, axis=1)

    elif method in ['lcmc', 'jaccard']:
        assert r_hd is not None, "r_hd must be provided for lcmc and jaccard distance"
        assert r_ld is not None, "r_ld must be provided for lcmc and jaccard distance"
        counts = np.array([len(np.intersect1d(a, b)) for a, b in zip(r_hd[:,1:K+1], r_ld[:,1:K+1])])
        if method == 'lcmc':
            return counts / K
        elif method == 'jaccard':
            counts_union = np.array([len(np.union1d(a, b)) for a, b in zip(r_hd[:,1:K+1], r_ld[:,1:K+1])])
            return counts / counts_union
    else:
        print('Invalid method')