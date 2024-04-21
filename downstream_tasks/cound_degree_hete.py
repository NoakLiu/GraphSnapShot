from sympy import degree, im
import torch
from ogb.nodeproppred import NodePropPredDataset
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Download and load the OGB-MAG dataset
dataset = NodePropPredDataset(name="ogbn-mag")

# Extract edge information
# edge_index = dataset[0].edge_index
# edge_index = dataset[0]['edge_index_dict']

# Assuming edge information is the second element (check library documentation)
edge_index = dataset[0][0]  # Access the second element (might differ based on library)
print(edge_index)

# Compute node degrees
edge_index_dict = edge_index['edge_index_dict']
edge_types = [('author', 'affiliated_with', 'institution'), ('author', 'writes', 'paper')]
edges = np.concatenate([edge_index_dict[edge_type] for edge_type in edge_types], axis=1)
print(edges)

# Flatten the array to get a single list of all involved nodes
all_nodes = np.concatenate((edges[0], edges[1]))

# Compute unique nodes and their degree (counts)
unique_nodes, degrees = np.unique(all_nodes, return_counts=True)

# Plotting the degree distribution
plt.figure(figsize=(10, 6))
# Using a log-log plot for better visualization of wide-range distributions
plt.loglog(range(len(degrees)), np.sort(degrees)[::-1], marker='o', linestyle='None')
plt.title('Degree Distribution')
plt.xlabel('Rank')
plt.ylabel('Degree')
plt.show()

# plotting a histogram of the degrees
plt.figure(figsize=(10, 6))
deg_hist, bins = np.histogram(degrees, bins=50)
plt.loglog(bins[:-1], deg_hist, 'bo-')
plt.title('Degree Distribution Histogram')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.show()