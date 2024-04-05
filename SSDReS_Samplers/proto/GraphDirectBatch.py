# 直接调用batch来取得adj信息，不考虑structure的情况

import random
import numpy as np

class GraphKSDSampler:
    def __init__(self, data, adj_matrix, N, k, n):
        self.data = data
        self.adj_matrix = adj_matrix
        self.N = N  # Total number of nodes to consider
        self.k = k  # Number of hops
        self.n = n  # Default number of nodes to sample in each batch

    def resample(self, n, n_per_hop=3):
        """
        Sample a batch of nodes and their corresponding k-hop adjacency matrices.
        """
        batch_nodes = random.sample(self.data, min(n, len(self.data)))
        self.batch_khop_matrices = self.compute_k_hop_matrices(batch_nodes, n_per_hop)

        return batch_nodes, self.batch_khop_matrices

    def compute_k_hop_matrices(self, nodes, n_per_hop):
        """
        Compute k-hop adjacency matrices for the given nodes.
        """
        khop_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

        # Start with the initial set of nodes
        current_layer_nodes = set(nodes)

        for i in range(self.k):
            next_layer_nodes = set()

            for node in current_layer_nodes:
                # Find neighbors, limit to n_per_hop if necessary
                neighbors = set(self.adj_matrix[node].nonzero()[0].tolist())
                if len(neighbors) > n_per_hop:
                    neighbors = set(random.sample(neighbors, n_per_hop))

                # Add edges to the k-hop matrix
                for neighbor in neighbors:
                    khop_matrices[i][node, neighbor] += 1
                    khop_matrices[i][neighbor, node] += 1  # For undirected graph
                    next_layer_nodes.add(neighbor)

            current_layer_nodes = next_layer_nodes

        return khop_matrices