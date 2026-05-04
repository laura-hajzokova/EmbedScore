# This script contains functions to compute various embeddings (UMAP, t-SNE, PCA, Diffusion Map, PHATE) for a given dataset.

# Import libraries
import pandas as pd
import numpy as np
import umap.umap_ as umap
import openTSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from pydiffmap import diffusion_map as dm
import phate
import matplotlib.pyplot as plt
import os
import vivae as vv

# Set the random seed for reproducibility
random_state = 42
path = f'data/'

# Compute the UMAP embedding
def compute_umap(data, labels=None, n_components=2):
    print('Computing the UMAP embedding...')
    model_umap = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_emb = model_umap.fit_transform(data)
    if labels is None:
        umap_emb_df = pd.DataFrame(umap_emb, columns=['umap1', 'umap2'])
    else:
        umap_emb_df = pd.DataFrame(np.concatenate((umap_emb,labels),axis=1), columns=['umap1', 'umap2','labels'])
    return umap_emb_df

# Compute the t-SNE embedding
def compute_tsne(data, labels=None, n_components=2, perplexity=30):
    print('Computing the t-SNE embedding...')
    #tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    #tsne_emb = tsne.fit_transform(data)
    aff = openTSNE.affinity.PerplexityBasedNN(data, perplexity=50, n_jobs=32, random_state=random_state)
    if os.path.exists('data/pca_emb.csv'):
        pca = pd.read_csv('data/pca_emb.csv')
        pca_emb = pca[['pc1', 'pc2']].values
        init = openTSNE.initialization.rescale(pca_emb)
        tsne_emb = openTSNE.TSNE(n_jobs=32,verbose=True).fit(affinities=aff, initialization=init)
    else:
        tsne_emb = openTSNE.TSNE(n_jobs=32,verbose=True).fit(affinities=aff)
    if labels is None:
        tsne_emb_df = pd.DataFrame(tsne_emb, columns=['tsne1', 'tsne2'])
    else:
        tsne_emb_df = pd.DataFrame(np.concatenate((tsne_emb,labels),axis=1), columns=['tsne1', 'tsne2','labels'])
    return tsne_emb_df
    
# Compute the PCA embedding    
def compute_pca(data, labels=None, n_components=2):
    print('Computing the PCA embedding...')
    pca = PCA(n_components=n_components)
    pca_emb = pca.fit_transform(np.array(data))
    if labels is None:
        pca_emb_df = pd.DataFrame(pca_emb, columns=['pc1', 'pc2'])
    else:
        pca_emb_df = pd.DataFrame(np.concatenate((pca_emb,labels),axis=1), columns=['pc1', 'pc2','labels'])
    return pca_emb_df

# Compute the Diffusion Map embedding
def compute_diffmap(data, labels=None, n_components=2):
    print('Computing the Diffusion Map embedding...')
    diffmap = dm.DiffusionMap.from_sklearn(n_evecs = n_components, epsilon = 1., alpha = 0.5, k=10)
    diffmap_emb = diffmap.fit_transform(data)
    if labels is None:
        diffmap_emb_df = pd.DataFrame(diffmap_emb, columns=['dm1', 'dm2'])
    else:
        diffmap_emb_df = pd.DataFrame(np.concatenate((diffmap_emb,labels),axis=1), columns=['dm1', 'dm2','labels'])
    return diffmap_emb_df

# Compute the PHATE embedding
def compute_phate(data, labels=None, n_components=2):
    print('Computing the PHATE embedding...')
    pht = phate.PHATE(n_components=n_components, n_jobs=-2, random_state=random_state)
    phate_emb = pht.fit_transform(data)
    if labels is None:
        phate_emb_df = pd.DataFrame(phate_emb, columns=['phate1', 'phate2'])
    else:
        phate_emb_df = pd.DataFrame(np.concatenate((phate_emb,labels),axis=1), columns=['phate1', 'phate2','labels'])
    return phate_emb_df

# Compute the ViVAE embedding
def compute_vivae(data, labels=None, n_components=2):
    print('Computing the ViVAE embedding...')
    model_vivae = vv.ViVAE(input_dim=data.shape[1], latent_dim=n_components)
    model_vivae.fit(data, n_epochs=100, lam_mds=10.)
    vivae_emb = model_vivae.transform(data)
    if labels is None:
        vivae_emb_df = pd.DataFrame(vivae_emb, columns=['vivae1', 'vivae2'])
    else:
        vivae_emb_df = pd.DataFrame(np.concatenate((vivae_emb,labels),axis=1), columns=['vivae1', 'vivae2','labels'])
    return vivae_emb_df

# Compute the Isomap embedding
def compute_isomap(data, labels=None, n_components=2):
    print('Computing the Isomap embedding...')
    isomap = Isomap(n_components=n_components, n_neighbors=10)
    isomap_emb = isomap.fit_transform(data)
    if labels is None:
        isomap_emb_df = pd.DataFrame(isomap_emb, columns=['isomap1', 'isomap2'])
    else:
        isomap_emb_df = pd.DataFrame(np.concatenate((isomap_emb,labels),axis=1), columns=['isomap1', 'isomap2','labels'])
    return isomap_emb_df

def plot_embedding(embedding, labels, title, emb_name=None):
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(title)
    if emb_name is not None:
        plt.xlabel(emb_name + ' 1')
        plt.ylabel(emb_name + ' 2')
    plt.colorbar(scatter, label='Cell type')
    plt.savefig(emb_name+'.png')
    plt.show()

## Get the embedding of the data
def get_embedding(data, labels=None, method='umap', n_components=2, perplexity=30):
    if type(data) != 'numpy.ndarray':
        data = np.array(data)
    if method == 'umap':
        if os.path.exists(os.path.join(path, 'umap_emb.csv')):
            emb = pd.read_csv(os.path.join(path, 'umap_emb.csv'))
            print('UMAP embedding loaded from file.')
        else:
            emb = compute_umap(data, labels, n_components)
            emb.to_csv(os.path.join(path, 'umap_emb.csv'), index=False)
            print('UMAP embedding saved.')
    elif method == 'tsne':
        if os.path.exists(os.path.join(path, 'tsne_emb.csv')):
            emb = pd.read_csv(os.path.join(path, 'tsne_emb.csv'))
            print('t-SNE embedding loaded from file.')
        else:
            emb = compute_tsne(data, labels, n_components, perplexity)
            emb.to_csv(os.path.join(path, 'tsne_emb.csv'), index=False)
            print('t-SNE embedding saved.')
    elif method == 'pca':
        if os.path.exists(os.path.join(path, 'pca_emb.csv')):
            emb = pd.read_csv(os.path.join(path, 'pca_emb.csv'))
            print('PCA embedding loaded from file.')
        else:
            emb = compute_pca(data, labels, n_components)
            emb.to_csv(os.path.join(path, 'pca_emb.csv'), index=False)
            print('PCA embedding saved.')
    elif method == 'diffmap':
        if os.path.exists(os.path.join(path, 'diffmap_emb.csv')):
            emb = pd.read_csv(os.path.join(path, 'diffmap_emb.csv'))
            print('Diffusion Map embedding loaded from file.')
        else:
            emb = compute_diffmap(data, labels, n_components)
            emb.to_csv(os.path.join(path, 'diffmap_emb.csv'), index=False)
            print('Diffusion Map embedding saved.')
    elif method == 'phate':
        if os.path.exists(os.path.join(path, 'phate_emb.csv')):
            emb = pd.read_csv(os.path.join(path, 'phate_emb.csv'))
            print('PHATE embedding loaded from file.')
        else:
            emb = compute_phate(data, labels, n_components)
            emb.to_csv(os.path.join(path, 'phate_emb.csv'), index=False)
            print('PHATE embedding saved.')
    elif method == 'vivae':
        if os.path.exists(os.path.join(path, 'vivae_emb.csv')):
            emb = pd.read_csv(os.path.join(path, 'vivae_emb.csv'))
            print('ViVAE embedding loaded from file.')
        else:
            emb = compute_vivae(data, labels, n_components)
            emb.to_csv(os.path.join(path, 'vivae_emb.csv'), index=False)
            print('ViVAE embedding saved.')
    elif method == 'isomap':
        if os.path.exists(os.path.join(path, 'isomap_emb.csv')):
            emb = pd.read_csv(os.path.join(path, 'isomap_emb.csv'))
            print('Isomap embedding loaded from file.')
        else:
             emb = compute_isomap(data, labels, n_components)
             emb.to_csv(os.path.join(path, 'isomap_emb.csv'), index=False)
             print('Isomap embedding saved.')
    else:
        raise ValueError(f'Invalid method: {method}')
    return emb