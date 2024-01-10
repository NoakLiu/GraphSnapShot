import random
import numpy as np
from Disk_Mem_Simulation.simulation_benchmark import benchmark

class GraphKSDSampler:
    def __init__(self, data, adj_matrix, N, k, n, batch_size):
        self.data = data
        self.adj_matrix = adj_matrix
        self.k = k
        self.n = n
        self.batch_size = batch_size
        self.static_sampled_nodes, self.k_hop_adjacency_matrices = self.preprocess_static_sampling(N)
        self.compute_matrices = [np.zeros((batch_size, *adj_matrix.shape), dtype=int) for _ in range(k)]

    def reset_compute_matrices(self):
        for matrix in self.compute_matrices:
            matrix.fill(0)

    def k_hop_sampling(self, initial_samples):
        layer_batches = [initial_samples]
        adjacency_matrix_batches = [np.zeros((self.batch_size, *self.adj_matrix.shape), dtype=int) for _ in range(self.k)]

        for i in range(self.k):
            next_layer_batch = [set() for _ in range(self.batch_size)]
            for batch_index, initial_sample in enumerate(layer_batches[-1]):
                current_layer = set(initial_sample)
                for node in current_layer:
                    benchmark.simulate_disk_read() if batch_index % 50 == 0 else None
                    neighbors = set(self.adj_matrix[node].nonzero()[0].tolist())
                    sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                    next_layer_batch[batch_index].update(sampled_neighbors)
                    benchmark.simulate_disk_write() if batch_index % 50 == 0 else None
                    for neighbor in sampled_neighbors:
                        adjacency_matrix_batches[i][batch_index, node, neighbor] += 1
                        adjacency_matrix_batches[i][batch_index, neighbor, node] += 1
            layer_batches.append([list(layer) for layer in next_layer_batch])

        return layer_batches, adjacency_matrix_batches

    def k_hop_retrieval(self, initial_samples):
        for batch_index, initial_sample in enumerate(initial_samples):
            current_layer = set(initial_sample)
            for i in range(self.k):
                next_layer = set()
                for node in current_layer:
                    benchmark.simulate_memory_access() if batch_index % 50 == 0 else None
                    neighbors = set(self.k_hop_adjacency_matrices[i][batch_index][node].nonzero()[0].tolist())
                    sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                    next_layer.update(sampled_neighbors)
                    benchmark.simulate_memory_access() if batch_index % 50 == 0 else None
                    for neighbor in sampled_neighbors:
                        self.k_hop_adjacency_matrices[i][batch_index][node, neighbor] -= 1
                        self.k_hop_adjacency_matrices[i][batch_index][neighbor, node] -= 1
                        self.compute_matrices[i][batch_index][node, neighbor] += 1
                        self.compute_matrices[i][batch_index][neighbor, node] += 1
                current_layer = next_layer

    def k_hop_resampling(self, initial_samples):
        for batch_index, initial_sample in enumerate(initial_samples):
            current_layer = set(initial_sample)
            for i in range(self.k):
                next_layer = set()
                for node in current_layer:
                    benchmark.simulate_disk_read() if batch_index % 50 == 0 else None
                    neighbors = set(self.adj_matrix[node].nonzero()[0].tolist())
                    sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                    next_layer.update(sampled_neighbors)
                    benchmark.simulate_disk_write() if batch_index % 50 == 0 else None
                    for neighbor in sampled_neighbors:
                        self.k_hop_adjacency_matrices[i][batch_index][node, neighbor] += 1
                        self.k_hop_adjacency_matrices[i][batch_index][neighbor, node] += 1
                current_layer = next_layer

    def preprocess_static_sampling(self, N):
        initial_samples = [random.sample(list(self.data), N) for _ in range(self.batch_size)]
        _, adjacency_matrices = self.k_hop_sampling(initial_samples)
        return initial_samples, adjacency_matrices

    def resample(self, n, alpha, n_per_hop=3):
        self.reset_compute_matrices()
        non_static_nodes = list(set(self.data) - set(sum(self.static_sampled_nodes, [])))

        dynamic_resampled_nodes = [random.sample(non_static_nodes, int((1 - alpha) * n)) for _ in range(self.batch_size)]
        static_resampled_nodes = [random.sample(nodes, int(alpha * n)) for nodes in self.static_sampled_nodes]

        self.k_hop_sampling(dynamic_resampled_nodes)
        self.k_hop_retrieval(static_resampled_nodes)

        combined_resampled_nodes = [dyn + stat for dyn, stat in zip(dynamic_resampled_nodes, static_resampled_nodes)]

        cutoff_dynamic_resampled_nodes = [random.sample(stat, int(alpha * n)) for stat in static_resampled_nodes]
        dynamic_resampled_nodes_swap_in = [random.sample(non_static_nodes, int(alpha * n)) for _ in range(self.batch_size)]
        self.k_hop_resampling(dynamic_resampled_nodes_swap_in)

        for i in range(self.batch_size):
            for node in cutoff_dynamic_resampled_nodes[i]:
                if node in self.static_sampled_nodes[i]:
                    self.static_sampled_nodes[i].remove(node)
            self.static_sampled_nodes[i].extend(dynamic_resampled_nodes_swap_in[i])

        return combined_resampled_nodes, self.compute_matrices

# Example usage:
# sampler = GraphKSDSampler(data, adj_matrix, N=100, k=5, n=10, batch_size=10)