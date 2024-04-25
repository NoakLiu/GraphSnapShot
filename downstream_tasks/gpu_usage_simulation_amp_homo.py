import numpy as np
import matplotlib.pyplot as plt
import torch
from ogb.nodeproppred import DglNodePropPredDataset

def gpu_usage_simulation(dataset_name, threshold, amp_rates, fanout):
    # Load the dataset
    dataset = DglNodePropPredDataset(name=dataset_name)
    graph, _ = dataset[0]
    degrees = graph.in_degrees() + graph.out_degrees()
    
    # Results storage
    results = []
    
    for amp_rate in amp_rates:
        # Nodes classification into sparse and dense
        sparse_mask = degrees < threshold
        dense_mask = degrees >= threshold
        
        # Calculate GPU usage for sparse graph (simply the count of edges connected to sparse nodes)
        sparse_edges = graph.subgraph(torch.where(sparse_mask)[0]).num_edges()
        
        # Resample the dense graph with amplified fanout
        dense_nodes = torch.where(dense_mask)[0].numpy()
        desired_sample_size = int(amp_rate * fanout) # Ensure not to exceed number of available nodes
        resampled_dense_edges = len(dense_nodes) * desired_sample_size
        # if desired_sample_size > 0:  # Check if there are enough nodes to sample
        #     resampled_dense_nodes = np.random.choice(dense_nodes, size=desired_sample_size, replace=False)
        #     resampled_dense_edges = graph.subgraph(torch.tensor(resampled_dense_nodes)).num_edges()
        # else:
        #     resampled_dense_edges = 0  # No nodes to sample, thus no edges
        
        # Calculate total GPU edge storage
        total_edges = sparse_edges + resampled_dense_edges
        results.append((amp_rate, sparse_edges, resampled_dense_edges, total_edges))
    
    return results

def plot_results(results):
    # Plotting the results
    amp_rates, sparse_usages, dense_usages, total_usages = zip(*results)
    plt.figure(figsize=(10, 5))
    plt.plot(amp_rates, sparse_usages, label='Sparse Graph Edges')
    plt.plot(amp_rates, dense_usages, label='Resampled Dense Graph Edges')
    plt.plot(amp_rates, total_usages, label='Total Edges')
    plt.xlabel('Amplication Rate')
    plt.ylabel('GPU Usage (Number of Edges)')
    plt.title('GPU Usage by Threshold and Amplification Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage Example
dataset_name = 'ogbn-arxiv' # products
threshold = 10
amp_rates = [1.5,2,2.5,3]
fanout = 10

results = gpu_usage_simulation(dataset_name, threshold, amp_rates, fanout)

print(results)
plot_results(results)
