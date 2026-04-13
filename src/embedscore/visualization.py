import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .compute_neighborhoods import get_neighbors, extract_neighbors_emb
from typing import Union

rng = np.random.default_rng(42)

def visualize_links(embedding: np.ndarray,
                    links: np.ndarray,
                    idcs: np.ndarray = None, 
                    threshold: float = 0.,
                    symmetric = True,
                    subsample_edges = False,
                    max_edges: int = 10000,
                    metric_name: str = 'Link quality', 
                    axes = None,
                    point_size: int = 4,
                    point_alpha: float = 0.5,
                    edge_width: float = 1.2,
                    edge_alpha: float = 0.5,
                    edge_cmap = 'coolwarm'):
    """
    Visualize network edges colored by quality metric
    
    Parameters:
    -----------
    embedding       - np.array (N,2), coordinates of points from the embedding
    links           - np.array (N,N), quality of the link between the points
    indices         - list or np.array (M,), indices of points to compute the links for
    threshold       - minimal link quality (for plotting)
    symmetric       - bool, if the link relation is symmetric or not
    subsample_edges - bool, subsample edges for plotting
    max_edges       - max number of edges (for plotting)
    metric_name     - str, name of the quality metric (for plotting)
    axes            - ax (for plotting)
    point_size      - size of the points (for plotting)
    point_alpha     - color transparency for points (for plotting)
    edge_width      - (for plotting)
    edge_alpha      - edge transparency (for plotting)
    edge_cmap       - str, continuous color map

    Return:
    -----------
    fig, ax
    """
    
    assert links.ndim == 2 and links.shape[0] == links.shape[1], "Distance matrix must be square"
    assert embedding.ndim == 2 and embedding.shape[1] == 2, "Embedding must be 2-dimensional."
    assert len(embedding) == len(links), "Size of the distance matrix and number of points must match."

    N = len(embedding)

    if idcs is not None:
        links = links[idcs,:]
    else:
        idcs = np.arange(N)
    
    rows, cols = np.nonzero(np.abs(links) > threshold)

    # Only upper triangle to avoid duplicates
    if symmetric:
        mask = rows < cols
        rows, cols = rows[mask], cols[mask]

    # Subsample edges if too many
    if subsample_edges and len(rows) > max_edges:
        edges = np.zeros((max_edges, 2, 2)) 
        edge_colors = np.zeros(max_edges)
        idcs_edges = rng.choice(len(rows), max_edges, replace=False)
        for idx, val in enumerate(idcs_edges):
            edges[idx] = [np.array(embedding[rows[val]]), np.array(embedding[cols[val]])]
            edge_colors[idx] = links[rows[val], cols[val]]
    else:
        edges = np.zeros((len(rows), 2, 2)) 
        edge_colors = np.zeros(len(rows))
        for idx, (i, j) in enumerate(zip(rows, cols)):
            edges[idx] = [np.array(embedding[i]), np.array(embedding[j])]
            edge_colors[idx] = links[i, j]

    # Create plot
    if axes is None:
        fig, ax = plt.subplots(figsize=(8,8))
    else:
        ax = axes
        fig = ax.figure
    
    # Draw points
    ax.scatter(np.array(embedding)[:, 0], np.array(embedding)[:, 1], 
                        c='lightgray', s=point_size, alpha=point_alpha*0.7, zorder=1)
    ax.scatter(np.array(embedding)[idcs, 0], np.array(embedding)[idcs, 1], 
                        c='black', s=point_size, alpha=point_alpha*0.7, zorder=3)

    if len(edges) > 0:
        # Draw edges
        lc = LineCollection(edges, linewidths=edge_width, alpha=edge_alpha, cmap=edge_cmap, zorder=2)
        lc.set_array(np.array(edge_colors))
        ax.add_collection(lc)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Link quality', rotation=270, labelpad=15)

    ax.set_title(f'{metric_name}')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if axes is None:
        return fig, ax
    else:
        return ax
    
def visualize_nodes(points: np.ndarray,                    
                    node_quality: np.ndarray,
                    sample_size=None,
                    axes=None,
                    metric_name: str = 'Node quality',
                    point_size=4,
                    point_alpha=0.3,
                    color_map='viridis'):
    """
    Visualize network nodes colored by quality metric

    Parameters:
    -----------
    points : array-like (N, 2)
        2D coordinates of points from the embedding
    vector : array-like (N,)
        Quality metric values for each point
    sample_size : int
        Number of points to sample (for performance)
    axes : matplotlib axes object
        Axes to plot on (default: create new figure)
    point_size : float
        Size of points in scatter plot
    point_alpha : float
        Alpha value for points in scatter plot
    """
    N = len(points)

    # Sample points if needed
    if sample_size is not None and sample_size < N:
        indices = rng.choice(N, sample_size, replace=False)
        points_sample = points[indices]
        vector_sample = node_quality[indices]
    else:
        points_sample = points
        vector_sample = node_quality
        indices = np.arange(N)

    # Create plot
    if axes is None:
        fig, ax = plt.subplots(figsize=(8,8))
    else:
        ax = axes
        fig = ax.figure 
    
    # Draw points
    sc = ax.scatter(points_sample[:, 0], points_sample[:, 1], 
                c=vector_sample, s=point_size, alpha=point_alpha, cmap='viridis', zorder=5)
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Node quality', rotation=270, labelpad=15)
    ax.set_title(f'{metric_name}')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()  

    if axes is None:
        return fig, ax
    else:
        return ax
    
def visualize_HDneighbours(embedding: np.ndarray,
                           D_hd: np.ndarray,
                           R_hd: np.ndarray,
                           K: int=100,
                           idcs_subset: Union[list,np.ndarray]=None,
                           ax=None):
    
    """
    Visualize HD neighbors in the LD embedding

    Parameters
    ----------
    embedding   - LD embedding coordinates, np.ndarray of shape (N, M) where usually M=2
    D           - np.ndarray of shape (N, N)
    R           - np.ndarray of shape (N, k), integer array where each row i
                contains the k column indices of the nearest neighbours for each point from D.
    K           - number of nearest neighbors
    idcs        - indices of the subsetted points

    Returns
    -------
    result  : np.ndarray of shape (N, 2); coordinates of the HD neighbors of the points from the subset.
    """

    N = len(D_hd)
    if idcs_subset is None:
        idcs_subset = np.arange(N)

    R_hd = get_neighbors(D_hd, K)[idcs_subset] 
   
    neighbors = extract_neighbors_emb(R_hd, idcs_subset)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(embedding[:, 0], embedding[:, 1], c='lightgray', s=2, alpha=0.5)
    ax.scatter(embedding[neighbors, 0], embedding[neighbors, 1], color='#2596be', label='HD neighbors', s=2, alpha=0.5)
    ax.scatter(embedding[idcs_subset, 0], embedding[idcs_subset, 1], c='black', s=2, alpha=0.9, label='Subset points')
    ax.set_aspect('equal')
    ax.set_title(f'HD neighbors (K={K}) of the subset of points')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, markerscale=3)
            
    return ax