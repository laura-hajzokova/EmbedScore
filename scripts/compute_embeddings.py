# Import libraries
import pandas as pd
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.embeddings import get_embedding

hd_data = np.load('data/Samusik_exprs.npy')
#hd_data = pd.read_csv("data/Samusik_exprs.npy")

### Compute the embeddings
## PCA
emb_pca = get_embedding(hd_data, method='pca', n_components=2)

## UMAP
emb_umap = get_embedding(hd_data, method='umap', n_components=2)

## t-SNE
emb_tsne = get_embedding(hd_data, method='tsne', n_components=2, perplexity=30)

## Diffusion Map
emb_diffmap = get_embedding(hd_data, method='diffmap', n_components=2)

## PHATE
emb_phate = get_embedding(hd_data, method='phate', n_components=2)

## ISOMAP
emb_isomap = get_embedding(hd_data, method='isomap', n_components=2)

## ViVAE
emb_vivae = get_embedding(hd_data, method='vivae', n_components=2)

print('Embeddings computed and saved successfully!')