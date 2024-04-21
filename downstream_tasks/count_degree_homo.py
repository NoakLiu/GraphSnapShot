import matplotlib.pyplot as plt
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset

# List of datasets to analyze
dataset_names = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']

for dataset_name in dataset_names:
    # Load dataset
    dataset = DglNodePropPredDataset(name=dataset_name)
    graph, _ = dataset[0]  # Note that tuple unpacking is used here to separate the graph and labels

    # Get node degrees
    deg = graph.in_degrees() + graph.out_degrees()
    print(deg)

    # Calculate degree distribution
    deg_hist = torch.bincount(deg)

    # Plot degree distribution
    plt.loglog(deg_hist.numpy(), 'bo-')
    plt.title('Degree Distribution for ' + dataset_name)
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')

    # Save the plot
    plt.savefig(dataset_name + '_degree_distribution.png')

    # Display the plot
    plt.show()
