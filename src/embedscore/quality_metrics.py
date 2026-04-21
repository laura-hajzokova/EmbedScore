from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy.linalg as LA   
from .compute_neighborhoods import get_neighbors, extract_neighbors_dist

rng = np.random.default_rng(42)

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

    assert method in ['intrusions', 'extrusions'], "Invalid method"

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

def link_qnx(D_hd: np.ndarray, D_ld: np.ndarray, r_hd: np.ndarray = None, r_ld: np.ndarray = None, K: int = 100, v: int = 1, w: int = 1, method: str = 'intrusions') -> np.ndarray:
    '''Computes links based on the Qnx criterion from [5]

    [5]   Lee, J. A. and Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria.

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

    assert method in ['intrusions', 'extrusions'], "Invalid method"

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

    if method == 'intrusions':
        links = np.maximum(R_hd-R_ld, np.zeros_like(R_hd)) ** v
        with np.errstate(divide='ignore', invalid='ignore'):
            norm1 = np.where(R_hd != 0, 1 / (R_hd**w), 0)  # 0 is the fallback value
        idx = r_ld[:,K+1:N]
    elif method == 'extrusions':
        links = np.maximum(R_ld-R_hd, np.zeros_like(R_hd)) ** v
        with np.errstate(divide='ignore', invalid='ignore'):
            norm1 = np.where(R_ld != 0, 1 / (R_ld**w), 0)  # 0 is the fallback value
        idx = r_hd[:,K+1:N]
      
    links =  links * norm1   # take the ranks of the points that are not in the `hd neighborhood`

    z = np.zeros((N,len(np.arange(K+1,N))))
    links[np.arange(N)[:, None], idx] = z
    
    return links

def link_distance_distortion(D_hd: np.ndarray, D_ld: np.ndarray, norm: float=None, method: str = 'compression', return_const: bool = False) -> np.ndarray:
    '''Computes links based on the Qnx criterion from [6]

    [6]   Aupetit, M. (2007). Visualizing distortions and recovering topology in continuous projection techniques.

    Parameters
        ----------
        D_hd    - numpy array (N,N), distance matrix of the original data
        D_ld    - numpy array (N,N), distance matrix of the embedding
        norm    - normalization factor
        method  - evaluation method, {'compression', 'stretching'}

    Return
        ------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings
    '''
    try:
        assert D_hd.shape == D_ld.shape, "Distance matrices must have equal shape"
    except AssertionError as e:
        print(f"Assertion failed: {e}")

    assert method in ['compression', 'stretching'], "Invalid method"

    N = len(D_hd)

    if method == 'compression':
        links = np.maximum(D_hd-D_ld, np.zeros_like(D_hd))
    elif method == 'stretching':
        links = - np.minimum(D_hd-D_ld, np.zeros_like(D_hd))

    if norm is None:
        norm = 1/(np.max(np.sum(links, axis=1)) - np.min(np.sum(links, axis=1)))
    
    links =  links * norm  
    
    if return_const:
        const = np.min(np.sum(links, axis=1)) * norm
        return links, const
    else:
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

    if method in ['trustworthiness', 'continuity', 'mrre','qnx']:
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
            return np.ones_like(counts) - (counts / counts_union)
    else:
        print('Invalid method')

def nodes_distance_distortion(links: np.ndarray, D_hd: np.ndarray=None, D_ld: np.ndarray=None, method: str = 'compression', const: float=None) -> np.ndarray:
    '''Computes nodes based on the distance distortion criterion

    Parameters      
        ----------
        links   - numpy array (N,N), quality of links between points in the original data and the embeddings
        D_hd    - numpy array (N,N), distance matrix of the original data
        D_ld    - numpy array (N,N), distance matrix of the embedding
        method  - evaluation method, {'compression', 'stretching'}
        const   - constant to subtract from the sum of links, if None, the sum of links is used

        Return
        ------
        nodes   - numpy array (N,), quality of nodes in the original data and the embeddings
    '''

    assert method in ['compression', 'stretching'], "Invalid method"

    if const is not None:
        nodes = np.sum(links, axis=1) - const*np.ones(links.shape[0])
    else:
        if D_hd is not None and D_ld is not None:
            assert D_hd.shape == D_ld.shape, "Distance matrices must have equal shape"
            if method == 'compression':
                norm_links = np.maximum(D_hd-D_ld, np.zeros_like(D_hd))
            elif method == 'stretching':
                norm_links = - np.minimum(D_hd-D_ld, np.zeros_like(D_hd))
            norm = np.min(np.sum(links, axis=1)) / (np.max(np.sum(links, axis=1)) - np.min(np.sum(links, axis=1)))
            nodes = np.sum(norm_links, axis=1) * norm
        else:
            print("Distance matrices must be provided if const is not provided")

    return nodes

def nodes_topographic_function(d_hd: np.ndarray, d_ld: np.ndarray, adj_hd: np.ndarray, adj_ld: np.ndarray, K: int=1):
    '''Computes nodes based on the topographic function

    Parameters      
        ----------
        d_hd    - numpy array (N,N), distance matrix of the original data given by the Delaunay triangulation
        d_ld    - numpy array (N,N), distance matrix of the embedding given by the Delaunay triangulation
        adj_hd  - numpy array (N,N), adjacency matrix of the original data on the Delaunay graph
        adj_ld  - numpy array (N,N), adjacency matrix of the embedding on the Delaunay graph
        K       - threshold for the topographic function

        Return
        ------
        nodes   - numpy array (N,), quality of nodes in the original data and the embeddings'''
    
    assert d_hd.shape == d_ld.shape == adj_hd.shape == adj_ld.shape, \
        "Distance and adjacency matrices must all have the same shape"
    
    def link_penalty(adj, d, threshold):
        return (adj.toarray() > 0) & (d > threshold)
   
    if K > 0:
        return np.sum(link_penalty(adj_hd, d_ld, K).astype(int), axis=1) 
    elif K < 0:
        return np.sum(link_penalty(adj_ld, d_hd, -K).astype(int), axis=1)
    else:
        return np.sum(link_penalty(adj_hd, d_ld, 1).astype(int) + link_penalty(adj_ld, d_hd, 1).astype(int), axis=1)