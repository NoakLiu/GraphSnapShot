import random
import numpy as np
# from Disk_Mem_Simulation.simulation_benchmark import benchmark
from Disk_Mem_Simulation.simulation_disk_cache_benchmark import benchmark

class GraphKSDSampler:
    def __init__(self, data, adj_matrix, k, batch_size, n, rcn, rdn, ecn):
        # super().__init__(data, N)  # Call to the __init__ of the super class
        self.data = data
        self.adj_matrix = adj_matrix
        self.k = k # k is the hop depth
        self.nodes_batch = batch_size
        self.cached_nodes_per_hop = n # n is the pre-sampled nodes per layer
        self.cached_retrieval_nodes_per_hop = rcn
        self.disk_retrieval_nodes_per_hop = rdn
        self.cached_erase_nodes_per_hop = ecn
        self.batch_base_nodes, self.k_hop_adjacency_matrices = self.preprocess_static_sampling()
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def preprocess_static_sampling(self):
        # --->这里最好每次是batch输入
        initial_sample = random.sample(list(self.data), self.nodes_batch)
        _, adjacency_matrices = self.k_hop_sampling(initial_sample)
        return initial_sample, adjacency_matrices

    def reset_compute_matrices(self):
        # Reset compute_matrices to zero matrices for new calculations
        self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def k_hop_sampling(self, initial_sample):
        layers = [initial_sample]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

        current_layer = set(initial_sample)

        cnt = len(current_layer)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:

                nonzero_indices = self.adj_matrix[node].nonzero()
                neighbors = set(nonzero_indices.reshape(-1).tolist())

                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.cached_nodes_per_hop))
                cnt += len(sampled_neighbors)
                next_layer.update(sampled_neighbors)

                for neighbor in sampled_neighbors:
                    adjacency_matrices[i][node, neighbor] += 1
                    adjacency_matrices[i][neighbor, node] += 1  # Assuming undirected graph

            layers.append(list(next_layer))
            current_layer = next_layer

        # load the structure embedding to the cache
        benchmark.simulate_disk_read(cnt)

        return layers, adjacency_matrices

    def k_hop_retrieval_cache(self):
        # Start with the initial sample set
        current_layer = set(self.batch_base_nodes)
        cnt = len(current_layer)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                # Retrieve neighbors from the i-th adjacency matrix
                neighbors = set(self.k_hop_adjacency_matrices[i][node].nonzero()[0].tolist())
                # neighbors = set(self.k_hop_adjacency_matrices[i][node].nonzero().reshape(-1).tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.cached_retrieval_nodes_per_hop))
                cnt += len(sampled_neighbors)
                next_layer.update(sampled_neighbors)

                for neighbor in sampled_neighbors:
                    # # Subtract the edge from the original adjacency matrix
                    # self.k_hop_adjacency_matrices[i][node, neighbor] -= 1
                    # self.k_hop_adjacency_matrices[i][neighbor, node] -= 1

                    # Add the edge to the compute_matrices for calculations
                    self.compute_matrices[i][node, neighbor] += 1
                    self.compute_matrices[i][neighbor, node] += 1

            current_layer = next_layer

        # access data from cache
        benchmark.simulate_cache_access(cnt)

    def k_hop_retrieval_disk(self):
        # Start with the initial sample set
        current_layer = set(self.batch_base_nodes)
        cnt = len(current_layer)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                nonzero_indices = self.adj_matrix[node].nonzero()
                neighbors = set(nonzero_indices.reshape(-1).tolist())

                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.disk_retrieval_nodes_per_hop))
                cnt += len(sampled_neighbors)
                next_layer.update(sampled_neighbors)

                for neighbor in sampled_neighbors:
                    # # Subtract the edge from the original adjacency matrix
                    # self.k_hop_adjacency_matrices[i][node, neighbor] -= 1
                    # self.k_hop_adjacency_matrices[i][neighbor, node] -= 1

                    # Add the edge to the compute_matrices for calculations
                    self.compute_matrices[i][node, neighbor] += 1
                    self.compute_matrices[i][neighbor, node] += 1

            current_layer = next_layer

        # access data from cache
        benchmark.simulate_cache_access(cnt)


    def k_hop_retrieval_erase(self):
        # Start with the initial sample set
        current_layer = set(self.batch_base_nodes)
        cnt = len(current_layer)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                # Retrieve neighbors from the i-th adjacency matrix
                neighbors = set(self.k_hop_adjacency_matrices[i][node].nonzero()[0].tolist())
                # neighbors = set(self.k_hop_adjacency_matrices[i][node].nonzero().reshape(-1).tolist())
                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.cached_erase_nodes_per_hop))
                cnt += len(sampled_neighbors)
                next_layer.update(sampled_neighbors)

                for neighbor in sampled_neighbors:
                    # Subtract the edge from the original adjacency matrix
                    self.k_hop_adjacency_matrices[i][node, neighbor] -= 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] -= 1

                    # # Add the edge to the compute_matrices for calculations
                    # self.compute_matrices[i][node, neighbor] += 1
                    # self.compute_matrices[i][neighbor, node] += 1

            current_layer = next_layer

        # erase data from cache
        benchmark.simulate_cache_access(cnt)

    def k_hop_resampling(self):
        layers = [self.batch_base_nodes]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

        current_layer = set(self.batch_base_nodes)
        cnt = len(current_layer)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:
                nonzero_indices = self.adj_matrix[node].nonzero()
                neighbors = set(nonzero_indices.reshape(-1).tolist())

                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.cached_nodes_per_hop))
                cnt += len(sampled_neighbors)
                next_layer.update(sampled_neighbors)

                for neighbor in sampled_neighbors:
                    self.k_hop_adjacency_matrices[i][node, neighbor] += 1
                    self.k_hop_adjacency_matrices[i][neighbor, node] += 1  # Assuming undirected graph

            layers.append(list(next_layer))
            current_layer = next_layer

        # load the structure embedding to the cache
        benchmark.simulate_disk_read(cnt)

    def resample(self, n, alpha, n_per_hop=3):
        self.reset_compute_matrices()

        # # non_static_nodes = list(set(list(self.adj_matrix[0])) - set(self.static_sampled_nodes))
        # non_static_nodes = list(set(list(self.data)) - set(self.batch_base_nodes))
        #
        # dynamic_resampled_nodes = random.sample(non_static_nodes, int((1 - alpha) * n))

        _, _ = self.k_hop_sampling(dynamic_resampled_nodes)  # , int((1-alpha) * n_per_hop)) # (1-alpha)*nph from disk --> just compute

        static_resampled_nodes = random.sample(self.batch_base_nodes, int(alpha * n))
        self.k_hop_retrieval_cache(static_resampled_nodes)  # , int(alpha * n_per_hop)) # alpha*nph from memory->remove

        combined_resampled_nodes = dynamic_resampled_nodes + static_resampled_nodes

        cutoff_dynamic_resampled_nodes = random.sample(static_resampled_nodes, int(alpha * n))

        dynamic_resampled_nodes_swap_in = random.sample(non_static_nodes, int(alpha * n))
        self.k_hop_resampling(
            dynamic_resampled_nodes_swap_in)  # , int(alpha * n_per_hop)) #alpha*nph from disk to memory

        for node in cutoff_dynamic_resampled_nodes:
            self.batch_base_nodes.remove(node)
        self.batch_base_nodes.extend(dynamic_resampled_nodes_swap_in)

        return combined_resampled_nodes, self.compute_matrices
