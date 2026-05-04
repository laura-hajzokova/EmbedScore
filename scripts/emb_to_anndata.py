import os
import pandas as pd
import numpy as np
import anndata as ad
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

embeddings = ['umap', 'tsne', 'pca', 'diffmap', 'phate', 'isomap']

# Load the embeddings
for emb in embeddings:
    filename = f"data/{emb}_emb.csv"
    if os.path.exists(filename):
        if emb == 'umap':
            emb_umap = pd.read_csv(filename)
        elif emb == 'tsne':
            emb_tsne = pd.read_csv(filename)
        elif emb == 'pca':
            emb_pca = pd.read_csv(filename)
        elif emb == 'diffmap':
            emb_diffmap = pd.read_csv(filename)
        elif emb == 'phate':
            emb_phate = pd.read_csv(filename)
        elif emb == 'isomap':
            emb_isomap = pd.read_csv(filename)
        else:
            print(f"Unknown embedding type: {emb}")

# Load metadata
#metadata = pd.read_csv("metadata.csv")
metadata = np.load('data/Samusik_labels.npy', allow_pickle=True)

labels = pd.DataFrame(metadata, columns=['cell_type'])

# Load the data
#hd_data = pd.read_csv("data/hd_data.csv")
hd_data = np.load('data/Samusik_exprs.npy')

# Create an AnnData object
d = ad.AnnData(
    X=hd_data.astype(np.float32),
    obs={
        'cell_type': np.asarray(labels['cell_type']).astype(np.str_)
    },
    obsm={
        'tSNE': np.asarray(emb_tsne).astype(np.float32),
        'UMAP': np.asarray(emb_umap).astype(np.float32),
        'PCA': np.asarray(emb_pca).astype(np.float32),
        'DiffMap': np.asarray(emb_diffmap).astype(np.float32),
        'PHATE': np.asarray(emb_phate).astype(np.float32),
        'Isomap': np.asarray(emb_isomap).astype(np.float32)
    },
    uns={
        'methods': {
            'tSNE': ['tSNE'],
            'UMAP': ['UMAP'],
            'PCA': ['PCA'],
            'Diffusion Maps': ['DiffMap'],
            'PHATE': ['PHATE'],
            'Isomap': ['Isomap']
        }
    }
)

# Save the AnnData object
d.write(filename='data/anndata_obj.h5ad')
print('Data saved successfully!')