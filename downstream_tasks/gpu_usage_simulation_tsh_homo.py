import numpy as np
import matplotlib.pyplot as plt
import torch
from ogb.nodeproppred import DglNodePropPredDataset

def gpu_usage_simulation(dataset_name, thresholds, amp_rate, fanout):
    # Load the dataset
    dataset = DglNodePropPredDataset(name=dataset_name)
    graph, _ = dataset[0]
    degrees = graph.in_degrees() + graph.out_degrees()
    
    # Results storage
    results = []
    
    for threshold in thresholds:
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
        results.append((threshold, sparse_edges, resampled_dense_edges, total_edges))
    
    return results

def plot_results(results):
    # Plotting the results
    thresholds, sparse_usages, dense_usages, total_usages = zip(*results)
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, sparse_usages, label='Sparse Graph Edges')
    plt.plot(thresholds, dense_usages, label='Resampled Dense Graph Edges')
    plt.plot(thresholds, total_usages, label='Total Edges')
    plt.xlabel('Degree Threshold')
    plt.ylabel('GPU Usage (Number of Edges)')
    plt.title('GPU Usage by Threshold and Amplification Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage Example
dataset_name = 'ogbn-arxiv'
thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
amp_rate = 1.5
fanout = 10

results = gpu_usage_simulation(dataset_name, thresholds, amp_rate, fanout)

print(results)
plot_results(results)
