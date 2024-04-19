import dgl
import torch

def split_graph_by_degree(graph, degree_threshold):
    """
    Splits a graph into two subgraphs based on whether nodes have a degree greater than a given threshold.
    
    Parameters:
    - graph (dgl.DGLGraph): The input DGL graph.
    - degree_threshold (int): The degree threshold.
    
    Returns:
    - high_degree_subgraph (dgl.DGLGraph): Subgraph containing nodes with a degree greater than the threshold.
    - low_degree_subgraph (dgl.DGLGraph): Subgraph containing nodes with a degree less than or equal to the threshold.
    """
    # Calculate degrees for each node
    degrees = graph.in_degrees()

    # Identify nodes with degrees greater than the threshold
    high_degree_nodes = torch.where(degrees > degree_threshold)[0]

    # Identify nodes with degrees less than or equal to the threshold
    low_degree_nodes = torch.where(degrees <= degree_threshold)[0]

    # Create subgraphs based on the identified nodes
    high_degree_subgraph = graph.subgraph(high_degree_nodes)
    low_degree_subgraph = graph.subgraph(low_degree_nodes)

    return high_degree_subgraph, low_degree_subgraph

from ogb.nodeproppred import DglNodePropPredDataset

def load_and_split_dataset(name, degree_threshold):
    """
    Loads an OGB dataset by name, converts it to an undirected graph, and splits it based on the degree threshold.
    
    Parameters:
    - name (str): Name of the OGB dataset to load.
    - degree_threshold (int): Degree threshold for splitting the graph.
    
    Returns:
    - high_degree_subgraph (dgl.DGLGraph): Subgraph of high-degree nodes.
    - low_degree_subgraph (dgl.DGLGraph): Subgraph of low-degree nodes.
    """
    dataset = DglNodePropPredDataset(name=name)
    graph, labels = dataset[0]  # Get the graph and labels
    # Convert to an undirected graph and remove multi-edges
    # ugraph = graph.to_simple()  
    high_degree_subgraph, low_degree_subgraph = split_graph_by_degree(graph, degree_threshold)
    return graph, high_degree_subgraph, low_degree_subgraph

# Example: Load 'ogbn-arxiv' and split it
graph, high_degree_subgraph, low_degree_subgraph = load_and_split_dataset('ogbn-arxiv', 30) # products

print(graph)

print(high_degree_subgraph)


print("High Degree Subgraph Nodes:", high_degree_subgraph.number_of_nodes())
print("Low Degree Subgraph Nodes:", low_degree_subgraph.number_of_nodes())



import dgl

def sample_neighbors_from_subgraph(subgraph, fanout):
    """
    Sample neighbors from a subgraph using DGL's sample_neighbors function.

    Parameters:
    - subgraph (dgl.DGLGraph): The subgraph from which to sample.
    - fanout (int): The number of neighbors to sample for each node.

    Returns:
    - sampled_subgraph (dgl.DGLGraph): The subgraph containing the sampled edges.
    """
    # Assume all nodes are involved in sampling
    all_nodes = subgraph.nodes()
    
    # Sample neighbors for all nodes
    sampled_subgraph = dgl.sampling.sample_neighbors(subgraph, all_nodes, fanout)
    
    return sampled_subgraph

# Assuming high_degree_subgraph is an already created subgraph
fanout = 10  # Sample 10 neighbors per node
sampled_high_degree_subgraph = sample_neighbors_from_subgraph(high_degree_subgraph, fanout)

print("Sampled Subgraph has", sampled_high_degree_subgraph.number_of_edges(), "edges")


