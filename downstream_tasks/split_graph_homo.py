import matplotlib.pyplot as plt
import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset

def plot_degree_distribution_with_threshold(dataset_name, thresholds=[10, 20, 30, 40, 50, 60]):
    # Load the dataset
    dataset = DglNodePropPredDataset(name=dataset_name)
    graph, _ = dataset[0]
    degrees = graph.in_degrees() + graph.out_degrees()
    
    # Calculate the degree histogram
    max_degree = degrees.max().item()
    degree_counts = np.bincount(degrees.numpy(), minlength=max_degree+1)

    fig, axs = plt.subplots(1, 6, figsize=(18, 3))  # Create subplots, one for each threshold
    dense_ratios = []
    sampling_ratios = []

    for i, threshold in enumerate(thresholds):
        # Plot the degree distribution histogram
        axs[i].bar(range(len(degree_counts)), degree_counts, color='blue', label='Degree Distribution')
        # Add a vertical line for the threshold
        axs[i].axvline(x=threshold, color='red', label=f'Threshold = {threshold}')
        axs[i].set_title(f'Threshold = {threshold}')
        axs[i].set_xlabel('Degree')
        axs[i].set_ylabel('Number of Nodes')
        axs[i].legend()

        # Calculate the proportion of dense graph
        dense_ratio = (degrees > threshold).float().mean().item()
        dense_ratios.append(dense_ratio)

        # Randomly sample 10% of the nodes
        sampled_nodes = np.random.choice(degrees.shape[0], size=int(0.1 * degrees.shape[0]), replace=False)
        # Calculate the proportion of dense graph in the sampled nodes
        sampled_dense_ratio = (degrees[sampled_nodes] > threshold).float().mean().item()
        sampling_ratios.append(sampled_dense_ratio)

    plt.tight_layout()
    plt.show()

    # Output results
    print("Dense graph ratios:")
    print(" ".join(f"{ratio:.2f}" for ratio in dense_ratios))
    print("Sampling in dense graph ratios:")
    print(" ".join(f"{ratio:.2f}" for ratio in sampling_ratios))

# Example usage
plot_degree_distribution_with_threshold('ogbn-arxiv')
