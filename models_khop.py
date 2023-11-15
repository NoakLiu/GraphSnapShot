import random
import numpy as np

from models import GraphSDSampler


class GraphKSDSampler(GraphSDSampler):
    def __init__(self, data, adj_matrix, N, k, n):
        super().__init__(data, N)  # Call to the __init__ of the super class
        self.data = data
        self.adj_matrix = adj_matrix
        self.k = k
        self.n = n
        self.static_sampled_nodes, self.k_hop_adjacency_matrices = self.preprocess_static_sampling(N)
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def reset_compute_matrices(self):
        # Reset compute_matrices to zero matrices for new calculations
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def k_hop_sampling(self, initial_sample):
        layers = [initial_sample]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

        current_layer = set(initial_sample)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                neighbors = set(self.adj_matrix[node].nonzero()[1].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                # Record edges in the adjacency matrix for the i-th hop
                for neighbor in sampled_neighbors:
                    adjacency_matrices[i][node, neighbor] += 1
                    adjacency_matrices[i][neighbor, node] += 1  # Assuming undirected graph

            layers.append(list(next_layer))
            current_layer = next_layer

        return layers, adjacency_matrices

    def k_hop_retrieval(self, initial_sample):
        # Start with the initial sample set
        current_layer = set(initial_sample)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                # Retrieve neighbors from the i-th adjacency matrix
                neighbors = set(self.k_hop_adjacency_matrices[i][node].nonzero()[1].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                # Update adjacency matrices
                for neighbor in sampled_neighbors:
                    # Subtract the edge from the original adjacency matrix
                    self.k_hop_adjacency_matrices[i][node, neighbor] -= 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] -= 1

                    # Add the edge to the compute_matrices for calculations
                    self.compute_matrices[i][node, neighbor] += 1
                    self.compute_matrices[i][neighbor, node] += 1

            current_layer = next_layer

    def k_hop_resampling(self, initial_sample):
        layers = [initial_sample]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

        current_layer = set(initial_sample)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                neighbors = set(self.adj_matrix[node].nonzero()[1].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                # Record edges in the adjacency matrix for the i-th hop
                for neighbor in sampled_neighbors:
                    self.k_hop_adjacency_matrices[i][node, neighbor] += 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] += 1  # Assuming undirected graph

            layers.append(list(next_layer))
            current_layer = next_layer

    def preprocess_static_sampling(self, N):
        initial_sample = random.sample(list(self.data), N)
        _, adjacency_matrices = self.k_hop_sampling(initial_sample)
        return initial_sample, adjacency_matrices

    def resample(self, n, alpha, n_per_hop=3):
        non_static_nodes = list(set(list(self.adj_matrix[0])) - set(self.static_sampled_nodes))

        dynamic_resampled_nodes = random.sample(non_static_nodes, int((1 - alpha) * n))
        _, _ = self.k_hop_sampling(
            dynamic_resampled_nodes)  # , int((1-alpha) * n_per_hop)) # (1-alpha)*nph from disk --> just compute

        static_resampled_nodes = random.sample(self.static_sampled_nodes, int(alpha * n))
        self.k_hop_retrieval(static_resampled_nodes)  # , int(alpha * n_per_hop)) # alpha*nph from memory->remove

        combined_resampled_nodes = dynamic_resampled_nodes + static_resampled_nodes

        cutoff_dynamic_resampled_nodes = random.sample(static_resampled_nodes, int(alpha * n))

        dynamic_resampled_nodes_swap_in = random.sample(non_static_nodes, int(alpha * n))
        self.k_hop_resampling(
            dynamic_resampled_nodes_swap_in)  # , int(alpha * n_per_hop)) #alpha*nph from disk to memory

        for node in cutoff_dynamic_resampled_nodes:
            self.static_sampled_nodes.remove(node)
        self.static_sampled_nodes.extend(dynamic_resampled_nodes_swap_in)

        return combined_resampled_nodes, self.compute_matrices