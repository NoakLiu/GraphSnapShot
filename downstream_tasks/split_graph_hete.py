import matplotlib.pyplot as plt
import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import dgl

def plot_degree_distribution_with_threshold(dataset_name, thresholds=[2, 3, 4, 5, 6, 7]):
    # Load the dataset
    dataset = DglNodePropPredDataset(name=dataset_name)
    graph, _ = dataset[0]
    
    # Calculate degrees for heterogeneous graph
    degrees = torch.zeros(graph.num_nodes())  # Initialize degrees to zero
    for etype in graph.canonical_etypes:
        src, dst = graph.edges(etype=etype)
        degrees[src] += 1
        degrees[dst] += 1

    # Calculate the degree histogram
    max_degree = int(degrees.max().item())
    degree_counts = np.bincount(degrees.numpy().astype(int), minlength=max_degree+1)

    fig, axs = plt.subplots(1, 6, figsize=(18, 3))  # Create subplots, one for each threshold
    dense_ratios = []
    sampling_ratios = []

    # Explanation of abbreviations
    fig.text(0.5, 0.0, 'DGR: Dense Graph Ratio, SDR: Sampling Dense Ratio', ha='center', va='bottom', fontsize=10)

    for i, threshold in enumerate(thresholds):
        # Plot the degree distribution histogram
        axs[i].bar(range(len(degree_counts)), degree_counts, color='blue', label='Degree Distribution')
        axs[i].axvline(x=threshold, color='red', label=f'Threshold = {threshold}')
        axs[i].set_title(f'Threshold = {threshold}')
        axs[i].set_xlabel('Degree')
        axs[i].set_ylabel('Number of Nodes')
        axs[i].legend()

        mean_degree = degrees.float().mean().item()
        median_degree = degrees.median().item()
        axs[i].text(0.95, 0.95, f'Mean: {mean_degree:.2f}\nMedian: {median_degree}', transform=axs[i].transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

        # Calculate the proportion of nodes above the threshold
        dense_ratio = (degrees > threshold).float().mean().item()
        dense_ratios.append(dense_ratio)

        # Randomly sample 10% of the nodes
        sampled_nodes = np.random.choice(graph.num_nodes(), size=int(0.1 * graph.num_nodes()), replace=False)
        sampled_dense_ratio = (degrees[sampled_nodes] > threshold).float().mean().item()
        sampling_ratios.append(sampled_dense_ratio)

        # Add text for ratios below the plots
        axs[i].text(0.7, 0.1, f'DGR: {dense_ratio:.2f}\nSDR: {sampled_dense_ratio:.2f}', transform=axs[i].transAxes, verticalalignment='bottom', horizontalalignment='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_degree_distributions_with_thresholds.png")
    plt.show()

    # Output results
    print("Dense graph ratios:")
    print(" ".join(f"{ratio:.2f}" for ratio in dense_ratios))
    print("Sampling in dense graph ratios:")
    print(" ".join(f"{ratio:.2f}" for ratio in sampling_ratios))

# Example usage
plot_degree_distribution_with_threshold('ogbn-mag')
