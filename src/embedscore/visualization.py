import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from .compute_neighborhoods import get_neighbors, extract_neighbors_emb
from typing import Union
import seaborn as sns

rng = np.random.default_rng(42)

def visualize_links(embedding: np.ndarray,
                    links: np.ndarray,
                    idcs: np.ndarray = None, 
                    threshold: float = 0.,
                    symmetric = False,
                    subsample_edges = False,
                    max_edges: int = 10000,
                    quantiles: bool = False,
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

    if quantiles:
        q_up = np.quantile(links, 0.95)
        q_low = np.quantile(links, 0.05)
        links = np.where(links > q_up, q_up, links)
        links = np.where(links < q_low, q_low, links)

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

    norm_col = Normalize(vmin=np.min(edges), vmax=np.max(edges))

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
        # Normalize edge_colors to [0, 1] for alpha mapping
        edge_colors_arr = np.array(edge_colors)
        norm = Normalize(vmin=edge_colors_arr.min(), vmax=edge_colors_arr.max())
        alphas = norm(edge_colors_arr)

        # Get RGBA from colormap, override alpha channel with value-based alpha
        cmap = plt.get_cmap(edge_cmap)
        colors = [(*cmap(norm(v))[:3], a) for v, a in zip(edge_colors_arr, alphas)]

        # Draw edges
        lc = LineCollection(edges, linewidths=edge_width, colors=colors, zorder=2)
        ax.add_collection(lc)

        # Add colorbar — needs a ScalarMappable since colors are now baked in
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
        sm.set_array(edge_colors_arr)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Link quality', rotation=270, labelpad=15)

    ax.set_title(f'{metric_name}')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

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
                    color_map='coolwarm'):
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
                c=vector_sample, s=point_size, alpha=point_alpha, cmap=color_map, zorder=5)
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Node quality', rotation=270, labelpad=15)
    ax.set_title(f'{metric_name}')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()  
    plt.show()

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
    plt.show()
            
    return ax

def plot_correlation_heatmap(corr_matrix: np.ndarray, 
                     labels: list, 
                     title: str = 'Correlation matrix', 
                     cmap: str = 'coolwarm', 
                     vmin: float = -1, 
                     vmax: float = 1, 
                     annot: bool = True, 
                     fmt: str = '.2f',
                     title_fontsize: int = 12,
                     tick_fontsize: int = 10,
                     annot_fontsize: int = 8,
                     ax=None):
    """
    Plot a correlation matrix as a heatmap.

    Parameters:
    -----------
    corr_matrix : np.ndarray
        Square matrix of shape (N, N) containing correlation values.
    labels : list
        List of length N containing labels for the axes.
    title : str
        Title of the plot.
    cmap : str
        Colormap to use for the heatmap.
    vmin : float
        Minimum value for colormap scaling.
    vmax : float
        Maximum value for colormap scaling.
    annot : bool
        Whether to annotate the heatmap with correlation values.
    fmt : str
        String format for annotations.
    title_fontsize : int
        Font size for the title.
    tick_fontsize : int
        Font size for the tick labels.
    annot_fontsize : int
        Font size for annotation text.

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        annot_kws={"size": annot_fontsize},
        xticklabels=labels,
        yticklabels=labels,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)
    plt.tight_layout()
    plt.show()

    return (fig, ax) if ax is None else ax

def plot_distributions(matrix: np.ndarray, quantiles: Union[np.ndarray]=None, title: str="Empirical PDF", ax_title: list=None,
                   bins: int=50, cols: int=3):
    """
    Plot a normalised histogram (empirical PDF) for each row of a matrix,
    one subplot per row. Much faster than KDE — suitable for large matrices.

    Parameters
    ----------
    matrix        : array-like, shape (N, K)
    quantiles     : if not None, values below this threshold are excluded before plotting
    title         : overall figure title
    ax_title      : list of titles for each subplot
    bins          : number of histogram bins (int) or a bin-edge array
    cols          : number of subplot columns
    """
   
    N = matrix.shape[0]
    assert (quantiles.shape[1] == N) and (quantiles.shape[0] == 2), "Quantiles should be a 2D array with shape (2, N) where the first row contains the lower quantiles and the second row contains the upper quantiles for each row of the matrix."

    cols = min(cols, N)
    rows = int(np.ceil(N / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, row in enumerate(matrix):
        data = row[(row > quantiles[0, i]) & (row < quantiles[1, i])] if quantiles is not None else row
        data = data[~np.isnan(data)]                     
        ax = axes[i]
        
        ax.hist(data, bins=bins, density=True,
                color="lightblue", alpha=0.75, edgecolor="none")
        if ax_title is not None and i < len(ax_title):
            ax.set_title(ax_title[i], fontsize=10)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide unused subplots
    for j in range(N, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    
    plt.show()
