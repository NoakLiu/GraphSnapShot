import random
import numpy as np
from Disk_Mem_Simulation.simulation_benchmark import benchmark
from IOCostOptimizer import IOCostOptimizer

class GraphKSDSampler:
    def __init__(self, data, adj_matrix, N, k, n, base_read_cost, base_write_cost):
        self.data = data
        self.adj_matrix = adj_matrix
        self.k = k
        self.n = n
        self.io_cost_optimizer = IOCostOptimizer(base_read_cost, base_write_cost)
        self.static_sampled_nodes, self.k_hop_adjacency_matrices = self.preprocess_static_sampling(N)
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def reset_compute_matrices(self):
        benchmark.simulate_memory_access()
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def k_hop_sampling(self, initial_sample):
        read_ops, write_ops = 0, 0
        layers = [initial_sample]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]
        current_layer = set(initial_sample)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                read_ops += 1
                benchmark.simulate_disk_read()
                neighbors = set(self.adj_matrix[node].nonzero()[0].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                write_ops += 1
                benchmark.simulate_disk_write()
                for neighbor in sampled_neighbors:
                    adjacency_matrices[i][node, neighbor] += 1
                    adjacency_matrices[i][neighbor, node] += 1

            layers.append(list(next_layer))
            current_layer = next_layer

        io_cost = self.io_cost_optimizer.estimate_query_cost(read_ops, write_ops)
        print(f"I/O cost for k_hop_sampling: {io_cost}")
        return layers, adjacency_matrices

    def k_hop_retrieval(self, initial_sample):
        read_ops, write_ops = 0, 0
        current_layer = set(initial_sample)
        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                read_ops += 1
                benchmark.simulate_memory_access()
                neighbors = set(self.k_hop_adjacency_matrices[i][node].nonzero()[0].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                write_ops += 1
                benchmark.simulate_memory_access()
                for neighbor in sampled_neighbors:
                    self.k_hop_adjacency_matrices[i][node, neighbor] -= 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] -= 1
                    self.compute_matrices[i][node, neighbor] += 1
                    self.compute_matrices[i][neighbor, node] += 1

            current_layer = next_layer

        io_cost = self.io_cost_optimizer.estimate_query_cost(read_ops, write_ops)
        print(f"I/O cost for k_hop_retrieval: {io_cost}")

    def k_hop_resampling(self, initial_sample):
        read_ops, write_ops = 0, 0
        layers = [initial_sample]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]
        current_layer = set(initial_sample)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                read_ops += 1
                benchmark.simulate_disk_read()
                neighbors = set(self.adj_matrix[node].nonzero()[0].tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.n))
                next_layer.update(sampled_neighbors)

                write_ops += 1
                benchmark.simulate_disk_write()
                for neighbor in sampled_neighbors:
                    self.k_hop_adjacency_matrices[i][node, neighbor] += 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] += 1

            layers.append(list(next_layer))
            current_layer = next_layer

        io_cost = self.io_cost_optimizer.estimate_query_cost(read_ops, write_ops)
        print(f"I/O cost for k_hop_resampling: {io_cost}")

    def preprocess_static_sampling(self, N):
        read_ops, write_ops = 0, 0
        initial_sample = random.sample(list(self.data), N)
        read_ops += len(initial_sample)
        layers, adjacency_matrices = self.k_hop_sampling(initial_sample)
        io_cost = self.io_cost_optimizer.estimate_query_cost(read_ops, write_ops)
        print(f"I/O cost for preprocess_static_sampling: {io_cost}")
        return initial_sample, adjacency_matrices

    def resample(self, n, alpha, n_per_hop=3):
        read_ops, write_ops = 0, 0
        self.reset_compute_matrices()
        non_static_nodes = list(set(list(self.data)) - set(self.static_sampled_nodes))
        read_ops += len(non_static_nodes)
        dynamic_resampled_nodes = random.sample(non_static_nodes, int((1 - alpha) * n))
        static_resampled_nodes = random.sample(self.static_sampled_nodes, int(alpha * n))
        _, _ = self.k_hop_sampling(dynamic_resampled_nodes)
        _, _ = self.k_hop_retrieval(static_resampled_nodes)
        combined_resampled_nodes = dynamic_resampled_nodes + static_resampled_nodes
        cutoff_dynamic_resampled_nodes = random.sample(static_resampled_nodes, int(alpha * n))
        dynamic_resampled_nodes_swap_in = random.sample(non_static_nodes, int(alpha * n))
        _, _ = self.k_hop_resampling(dynamic_resampled_nodes_swap_in)
        for node in cutoff_dynamic_resampled_nodes:
            self.static_sampled_nodes.remove(node)
        self.static_sampled_nodes.extend(dynamic_resampled_nodes_swap_in)
        io_cost = self.io_cost_optimizer.estimate_query_cost(read_ops, write_ops)
        print(f"I/O cost for resample: {io_cost}")
        return combined_resampled_nodes, self.compute_matrices

# Example usage
sampler = GraphKSDSampler(data, adj_matrix, N, k, n, base_read_cost=0.05, base_write_cost=0.1)
