import dgl
import torch

def calculate_total_degree(graph, node_type):
    """
    Calculates the total degree (sum of in-degrees and out-degrees) for a specific node type in a heterogeneous graph.
    This function iterates over all edge types and sums up in-degrees and out-degrees for the specified node type.
    """
    total_degrees = torch.zeros(graph.number_of_nodes(node_type))  # Initialize a tensor to store total degrees

    # Iterate over all canonical edge types
    for etype in graph.canonical_etypes:
        src_type, _, dst_type = etype
        # If the node type is either the source or destination in the edge type
        if dst_type == node_type:
            total_degrees += graph.in_degrees(etype=etype)  # Add in-degrees
        if src_type == node_type:
            total_degrees += graph.out_degrees(etype=etype)  # Add out-degrees

    return total_degrees

def split_graph_by_total_degree(graph, node_type, degree_threshold):
    """
    Splits a graph into two subgraphs based on a total degree threshold for a specific node type.
    Nodes with a total degree above the threshold will be in one subgraph, and those with a total degree below or equal in another.
    """
    total_degrees = calculate_total_degree(graph, node_type)
    high_degree_nodes = torch.where(total_degrees > degree_threshold)[0]  # Nodes above the threshold
    low_degree_nodes = torch.where(total_degrees <= degree_threshold)[0]  # Nodes at or below the threshold

    # Create subgraphs using the identified nodes
    high_degree_subgraph = graph.subgraph({node_type: high_degree_nodes})
    low_degree_subgraph = graph.subgraph({node_type: low_degree_nodes})

    return high_degree_subgraph, low_degree_subgraph

# Example usage
from ogb.nodeproppred import DglNodePropPredDataset

# Load dataset
dataset = DglNodePropPredDataset(name='ogbn-mag')
graph, labels = dataset[0]  # graph is a heterogeneous graph

# graph = graph.to_simple()


# Degree threshold for splitting
degree_threshold = 30

# Split the graph based on the total degree of 'paper' nodes
high_degree_subgraph, low_degree_subgraph = split_graph_by_total_degree(graph, 'paper', degree_threshold)

print("High Degree Subgraph Nodes:", high_degree_subgraph.number_of_nodes())
print("Low Degree Subgraph Nodes:", low_degree_subgraph.number_of_nodes())
