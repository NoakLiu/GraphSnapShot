import random
import numpy as np
from Disk_Mem_Simulation.simulation_benchmark import benchmark
from BufferManager import BufferManager

class GraphKSDSampler:
    def __init__(self, data, adj_matrix, N, k, n, buffer_capacity=10):
        self.data = data
        self.adj_matrix = adj_matrix
        self.k = k
        self.n = n
        self.buffer_manager = BufferManager(capacity=buffer_capacity)
        self.static_sampled_nodes, self.k_hop_adjacency_matrices = self.preprocess_static_sampling(N)
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def reset_compute_matrices(self):
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def k_hop_sampling(self, initial_sample):
        buffer_key = f'k_hop_sampling_{tuple(initial_sample)}'
        cached_result = self.buffer_manager.get_data(buffer_key)
        if cached_result:
            return cached_result

        layers = [initial_sample]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]
        current_layer = set(initial_sample)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                if i % 50 == 0:
                    benchmark.simulate_disk_read()
                neighbors = set(self.adj_matrix[node].nonzero()[0].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                if i % 50 == 0:
                    benchmark.simulate_disk_write()
                for neighbor in sampled_neighbors:
                    adjacency_matrices[i][node, neighbor] += 1
                    adjacency_matrices[i][neighbor, node] += 1

            layers.append(list(next_layer))
            current_layer = next_layer

        self.buffer_manager.store_data(buffer_key, (layers, adjacency_matrices))
        return layers, adjacency_matrices

    def k_hop_retrieval(self, initial_sample):
        buffer_key = f'k_hop_retrieval_{tuple(initial_sample)}'
        cached_result = self.buffer_manager.get_data(buffer_key)
        if cached_result:
            return cached_result

        current_layer = set(initial_sample)
        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                if i % 50 == 0:
                    benchmark.simulate_memory_access()
                neighbors = set(self.k_hop_adjacency_matrices[i][node].nonzero()[0].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                if i % 50 == 0:
                    benchmark.simulate_memory_access()
                for neighbor in sampled_neighbors:
                    self.k_hop_adjacency_matrices[i][node, neighbor] -= 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] -= 1
                    self.compute_matrices[i][node, neighbor] += 1
                    self.compute_matrices[i][neighbor, node] += 1

            current_layer = next_layer

        self.buffer_manager.store_data(buffer_key, (self.k_hop_adjacency_matrices, self.compute_matrices))
        return self.k_hop_adjacency_matrices, self.compute_matrices

    def k_hop_resampling(self, initial_sample):
        buffer_key = f'k_hop_resampling_{tuple(initial_sample)}'
        cached_result = self.buffer_manager.get_data(buffer_key)
        if cached_result:
            return cached_result

        layers = [initial_sample]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]
        current_layer = set(initial_sample)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                if i % 50 == 0:
                    benchmark.simulate_disk_read()
                neighbors = set(self.adj_matrix[node].nonzero()[0].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                if i % 50 == 0:
                    benchmark.simulate_disk_write()
                for neighbor in sampled_neighbors:
                    self.k_hop_adjacency_matrices[i][node, neighbor] += 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] += 1

            layers.append(list(next_layer))
            current_layer = next_layer

        self.buffer_manager.store_data(buffer_key, (layers, self.k_hop_adjacency_matrices))
        return layers, self.k_hop_adjacency_matrices

    def preprocess_static_sampling(self, N):
        buffer_key = f'preprocess_static_sampling_{N}'
        cached_result = self.buffer_manager.get_data(buffer_key)
        if cached_result:
            return cached_result

        initial_sample = random.sample(list(self.data), N)
        layers, adjacency_matrices = self.k_hop_sampling(initial_sample)
        self.buffer_manager.store_data(buffer_key, (initial_sample, adjacency_matrices))
        return initial_sample, adjacency_matrices

    def resample(self, n, alpha, n_per_hop=3):
        buffer_key = f'resample_{n}_{alpha}_{n_per_hop}'
        cached_result = self.buffer_manager.get_data(buffer_key)
        if cached_result:
            return cached_result

        self.reset_compute_matrices()
        non_static_nodes = list(set(list(self.data)) - set(self.static_sampled_nodes))
        dynamic_resampled_nodes = random.sample(non_static_nodes, int((1 - alpha) * n))
        self.k_hop_sampling(dynamic_resampled_nodes)
        static_resampled_nodes = random.sample(self.static_sampled_nodes, int(alpha * n))
        self.k_hop_retrieval(static_resampled_nodes)
        combined_resampled_nodes = dynamic_resampled_nodes + static_resampled_nodes
        cutoff_dynamic_resampled_nodes = random.sample(static_resampled_nodes, int(alpha * n))
        dynamic_resampled_nodes_swap_in = random.sample(non_static_nodes, int(alpha * n))
        self.k_hop_resampling(dynamic_resampled_nodes_swap_in)

        for node in cutoff_dynamic_resampled_nodes:
            self.static_sampled_nodes.remove(node)
        self.static_sampled_nodes.extend(dynamic_resampled_nodes_swap_in)

        self.buffer_manager.store_data(buffer_key, (combined_resampled_nodes, self.compute_matrices))
        return combined_resampled_nodes, self.compute_matrices
