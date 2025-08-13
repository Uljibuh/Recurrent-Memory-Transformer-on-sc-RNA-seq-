#A. Genomic-position order 
#What: Sort genes by genomic coordinates (chr → start → strand).
#Why: Nearby genes share regulatory context (TADs, enhancers, bidirectional promoters). A local attention bias along this order can learn cis-effects efficiently.

#Co-expression order
#What: Order genes so neighbors are co-expressed across cells.
#Why: Lets local attention capture gene programs/modules even if they’re on different chromosomes.

import torch
import numpy as np
import pandas as pd

# Step 1: Simulate toy scRNA-seq data
# We’ll pretend we have 6 “genes” and 4 “cells.”

import torch
import numpy as np
import pandas as pd

# Toy gene expression matrix (cells x genes)
# Normally this would be counts, here just small integers
X = torch.tensor([
    [5, 0, 3, 0, 2, 1],
    [4, 1, 0, 2, 0, 3],
    [0, 2, 4, 1, 3, 0],
    [1, 3, 0, 4, 0, 2]
], dtype=torch.float32)

genes = ["G1", "G2", "G3", "G4", "G5", "G6"]
print(pd.DataFrame(X.numpy(), columns=genes))

#Step 2: Reorder by genomic position
#Let’s say we know genomic positions for each gene (totally fake data):

# Fake genomic positions (lower number = earlier position on chromosome)
genomic_positions = {
    "G1": 300,
    "G2": 100,
    "G3": 250,
    "G4": 400,
    "G5": 350,
    "G6": 150
}

# Sort genes by genomic position
genes_genomic_order = sorted(genes, key=lambda g: genomic_positions[g])
print("Genomic order:", genes_genomic_order)

# Reorder X accordingly
idx_genomic_order = [genes.index(g) for g in genes_genomic_order]
X_genomic = X[:, idx_genomic_order]


#Step 3: Reorder by co-expression pattern

#We’ll compute the average expression across cells and sort by descending mean.


gene_means = X.mean(dim=0)  # mean across cells
genes_coexp_order = [g for _, g in sorted(zip(gene_means, genes), reverse=True)]
print("Co-expression order:", genes_coexp_order)

idx_coexp_order = [genes.index(g) for g in genes_coexp_order]
X_coexp = X[:, idx_coexp_order]



#Step 4: Combine genomic & co-expression ordering
#One simple way:

#Normalize genomic positions to [0,1]

#Normalize mean expressions to [0,1]

#Take weighted sum and sort by that.
  
from sklearn.preprocessing import MinMaxScaler

pos_vals = np.array([genomic_positions[g] for g in genes]).reshape(-1,1)
mean_vals = gene_means.numpy().reshape(-1,1)

scaler = MinMaxScaler()
pos_norm = scaler.fit_transform(pos_vals).flatten()
mean_norm = scaler.fit_transform(mean_vals).flatten()

alpha = 0.5  # weight for genomic position
beta = 0.5   # weight for co-expression
combined_score = alpha*(1 - pos_norm) + beta*mean_norm  # smaller position is better

genes_combined_order = [g for _, g in sorted(zip(combined_score, genes), reverse=True)]
print("Combined order:", genes_combined_order)

idx_combined_order = [genes.index(g) for g in genes_combined_order]
X_combined = X[:, idx_combined_order]


#Step 5: Precompute bias matrices

#In transformers, we can add bias terms to attention scores to encourage certain patterns.
n_genes = len(genes)

# Genomic distance bias: smaller distance = larger bias
pos_array = np.array([genomic_positions[g] for g in genes_combined_order])
genomic_bias = -np.abs(pos_array[:, None] - pos_array[None, :]) / 1000.0  # scale
genomic_bias = torch.tensor(genomic_bias, dtype=torch.float32)

# Co-expression bias: correlation matrix
coexp_matrix = np.corrcoef(X_combined.T.numpy())  # genes x genes
coexp_bias = torch.tensor(coexp_matrix, dtype=torch.float32)

print("Genomic bias matrix:\n", genomic_bias)
print("Coexpression bias matrix:\n", coexp_bias)
