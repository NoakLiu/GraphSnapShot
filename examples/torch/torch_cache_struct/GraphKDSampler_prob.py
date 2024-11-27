import random
import numpy as np
import torch
from Disk_Mem_Simulation.simulation_benchmark import benchmark

class GraphKSDSampler:
    def __init__(self, data, adj_matrix, N, k, n):
        self.data = data
        self.adj_matrix = adj_matrix
        self.k = k
        self.n = n
        self.static_sampled_nodes, self.k_hop_adjacency_matrices = self.preprocess_static_sampling(N)
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]
        # self.node_degrees = np.sum(self.adj_matrix, axis=1)  # Calculate node degrees for probability-based sampling

        # self.node_degrees = torch.sum(self.adj_matrix, dim=1)

        self.node_degrees = torch.sum(adj_matrix, dim=1).float()  # Calculate node degrees
        self.node_probabilities = self.node_degrees / torch.sum(
            self.node_degrees)  # Normalize for probability distribution

    def probability_sampling(self, neighbors, n_samples):
        if len(neighbors) <= n_samples:
            return neighbors
        probabilities = self.node_probabilities[neighbors]
        sampled_indices = torch.multinomial(probabilities, n_samples, replacement=False)
        return neighbors[sampled_indices].tolist()

    def reset_compute_matrices(self):
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    # def probability_sampling(self, neighbors, n_samples):
    #     if len(neighbors) <= n_samples:
    #         return neighbors
    #     probabilities = [self.node_degrees[neighbor] for neighbor in neighbors]
    #     probabilities /= np.sum(probabilities)
    #     return np.random.choice(neighbors, n_samples, replace=False, p=probabilities).tolist()

    def k_hop_presampling(self, initial_sample):
        all_layers_node_set = set(initial_sample)
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

        current_layer = set(initial_sample)
        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                nonzero_indices = self.adj_matrix[node].nonzero()[0]
                neighbors = set(nonzero_indices.tolist())
                sampled_neighbors = self.probability_sampling(neighbors, self.n)
                next_layer.update(sampled_neighbors)
                for neighbor in sampled_neighbors:
                    adjacency_matrices[i][node, neighbor] = 1
                    adjacency_matrices[i][neighbor, node] = 1  # For undirected graph

            current_layer = next_layer

        return adjacency_matrices

    def k_hop_retrieval(self, initial_sample):
        current_layer = set(initial_sample)
        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                neighbors = set(self.k_hop_adjacency_matrices[i][node].nonzero()[0].tolist())
                sampled_neighbors = self.probability_sampling(neighbors, self.n)
                next_layer.update(sampled_neighbors)
                for neighbor in sampled_neighbors:
                    self.k_hop_adjacency_matrices[i][node, neighbor] -= 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] -= 1
                    self.compute_matrices[i][node, neighbor] += 1
                    self.compute_matrices[i][neighbor, node] += 1

            current_layer = next_layer

    def k_hop_resampling(self, initial_sample):
        all_layers_node_set = set(initial_sample)
        layers = [initial_sample]
        current_layer = set(initial_sample)
        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                nonzero_indices = self.adj_matrix[node].nonzero()[0]
                neighbors = set(nonzero_indices.tolist())
                sampled_neighbors = self.probability_sampling(neighbors, self.n)
                next_layer.update(sampled_neighbors)
                for neighbor in sampled_neighbors:
                    self.k_hop_adjacency_matrices[i][node, neighbor] += 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] += 1

            layers.append(list(next_layer))
            current_layer = next_layer
            all_layers_node_set.update(next_layer)

        all_layers_node_set = list(all_layers_node_set)
        return all_layers_node_set

    def preprocess_static_sampling(self, N):
        initial_sample = random.sample(list(self.data), N)
        adjacency_matrices = self.k_hop_presampling(initial_sample)
        return initial_sample, adjacency_matrices

    def resample(self, n, alpha, n_per_hop=3):
        self.reset_compute_matrices()
        non_static_nodes = list(set(self.data) - set(self.static_sampled_nodes))

        dynamic_resampled_nodes = random.sample(non_static_nodes, int((1 - alpha) * n))
        dynamic_resampled_nodes_struct = self.k_hop_resampling(dynamic_resampled_nodes)

        static_resampled_nodes = random.sample(self.static_sampled_nodes, int(alpha * n))
        self.k_hop_retrieval(static_resampled_nodes)

        combined_resampled_nodes = dynamic_resampled_nodes + static_resampled_nodes

        cutoff_dynamic_resampled_nodes = random.sample(static_resampled_nodes, int(alpha * n))

        dynamic_resampled_nodes_swap_in = random.sample(non_static_nodes, int(alpha * n))
        self.k_hop_resampling(dynamic_resampled_nodes_swap_in)

        for node in cutoff_dynamic_resampled_nodes:
            self.static_sampled_nodes.remove(node)
        self.static_sampled_nodes.extend(dynamic_resampled_nodes_swap_in)

        return combined_resampled_nodes, self.compute_matrices
