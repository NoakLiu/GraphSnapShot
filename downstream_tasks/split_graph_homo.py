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

    # Explanation of abbreviations
    fig.text(0.5, 0.0, 'DGR: Dense Graph Ratio, SDR: Sampling Dense Ratio', ha='center', va='bottom', fontsize=10)

    for i, threshold in enumerate(thresholds):
        # Calculate values for log-log plot
        x = np.arange(len(degree_counts))[degree_counts > 0]
        y = degree_counts[degree_counts > 0]

        # Plot the degree distribution histogram on log-log scale
        axs[i].loglog(x, y, 'bo-', label='Degree Distribution')
        # Add a vertical line for the threshold
        axs[i].axvline(x=threshold, color='red', label=f'Threshold = {threshold}')
        axs[i].set_title(f'Threshold = {threshold}')
        axs[i].set_xlabel('Degree')
        axs[i].set_ylabel('Number of Nodes')
        axs[i].legend()

        # Calculate and display mean and median degree
        mean_degree = degrees.float().mean().item()
        median_degree = degrees.median().item()
        axs[i].text(0.95, 0.95, f'Mean: {mean_degree:.2f}\nMedian: {median_degree}', transform=axs[i].transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

        # Calculate the proportion of nodes above the threshold
        dense_ratio = (degrees > threshold).float().mean().item()
        dense_ratios.append(dense_ratio)

        # Randomly sample 10% of the nodes
        sampled_nodes = np.random.choice(degrees.shape[0], size=int(0.1 * degrees.shape[0]), replace=False)
        sampled_dense_ratio = (degrees[sampled_nodes] > threshold).float().mean().item()
        sampling_ratios.append(sampled_dense_ratio)

        # Add text for ratios
        axs[i].text(0.7, 0.3, f'DGR: {dense_ratio:.2f}\nSDR: {sampled_dense_ratio:.2f}', transform=axs[i].transAxes, verticalalignment='bottom', horizontalalignment='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Save the entire figure
    plt.savefig(f"{dataset_name}_degree_distributions.png")

    plt.show()

# Example usage
plot_degree_distribution_with_threshold('ogbn-products') #arxiv
