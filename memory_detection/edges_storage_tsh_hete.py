import numpy as np
import matplotlib.pyplot as plt
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

def gpu_usage_simulation(dataset_name, thresholds, amp_rate, fanout):
    # Load the dataset
    dataset = DglNodePropPredDataset(name=dataset_name)
    graph, label = dataset[0]  # This will load the graph
    
    # Assuming we're focusing on 'paper' nodes
    # Check and extract node type
    if 'paper' in graph.ntypes:
        node_type = 'paper'
        # Calculate degrees only for 'paper' nodes
        degrees = graph.in_degrees(etype=('paper', 'cites', 'paper')) + graph.out_degrees(etype=('paper', 'cites', 'paper'))
        baseline = graph.number_of_edges(etype=('paper', 'cites', 'paper'))
    else:
        raise Exception("Specified node type 'paper' not found in graph")

    # Results storage
    results = []
    
    for threshold in thresholds:
        # Nodes classification into sparse and dense
        sparse_mask = degrees < threshold
        dense_mask = degrees >= threshold
        
        # Create subgraphs for sparse and dense nodes correctly handling heterogeneous graphs
        sparse_nodes = {node_type: torch.where(sparse_mask)[0]}
        dense_nodes = {node_type: torch.where(dense_mask)[0]}

        # Calculate GPU usage for sparse graph
        sparse_subgraph = graph.subgraph(sparse_nodes)
        sparse_edges = sparse_subgraph.num_edges()
        
        # Resample the dense graph with amplified fanout
        # For simplicity, just calculate based on the number of nodes
        desired_sample_size = int(amp_rate * fanout)
        resampled_dense_edges = len(dense_nodes[node_type]) * desired_sample_size
        
        # Calculate total GPU edge storage
        total_edges = sparse_edges + resampled_dense_edges
        results.append((threshold, sparse_edges, resampled_dense_edges, total_edges))
    
    return results, baseline

def plot_results(results, dataset_name):
    # Unpack the results
    thresholds, sparse_usages, dense_usages, total_usages = zip(*results)

    # Set up the plot
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, sparse_usages, label='Sparse Graph Storage')
    plt.plot(thresholds, dense_usages, label='Resampled Dense Graph Storage')
    plt.plot(thresholds, total_usages, label='Total Storage')

    # Find and mark the minimum total storage
    min_index = np.argmin(total_usages)
    min_threshold = thresholds[min_index]
    min_total_usage = total_usages[min_index]
    plt.scatter([min_threshold], [min_total_usage], color='red', s=100, label='Minimum Total Storage', zorder=5)

    # Configure plot aesthetics
    plt.xlabel('Degree Threshold')
    plt.ylabel('GPU Usage (Number of Edges)')
    plt.title(f'GPU Usage by Threshold for {dataset_name}')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Example Usage
dataset_name = 'ogbn-mag'  # Adjust the dataset name as necessary
thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
amp_rate = 1.5
fanout = 10

results, baseline = gpu_usage_simulation(dataset_name, thresholds, amp_rate, fanout)

print(results)
print(baseline)
# plot_results(results, dataset_name)

# baseline ogbn-mag 5416271
