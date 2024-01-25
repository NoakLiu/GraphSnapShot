import random
import numpy as np
from Disk_Mem_Simulation.simulation_disk_cache_benchmark import benchmark

class GraphFBLLoader:
    def __init__(self, data, adj_matrix, k, batch_size, drn):
        self.data = data
        self.adj_matrix = adj_matrix
        self.k = k # k is the hop depth
        self.nodes_batch = batch_size
        self.disk_read_nodes_per_hop = drn
        # self.cached_nodes_per_hop = n # n is the pre-sampled nodes per layer
        # self.cached_retrieval_nodes_per_hop = rcn
        # self.disk_retrieval_nodes_per_hop = rdn
        # self.cached_erase_nodes_per_hop = ecn
        self.batch_base_nodes, self.compute_matrices = self.setup_initial_sample_nodes()
        # self.compute_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

    def setup_initial_sample_nodes(self):
        # --->这里最好每次是batch输入
        initial_sample = random.sample(list(self.data), self.nodes_batch)
        _, adjacency_matrices = self.k_hop_sample_from_disk(initial_sample)
        return initial_sample, adjacency_matrices
    def k_hop_sample_from_disk(self, initial_sample):
        layers = [initial_sample]
        adjacency_matrices = [np.zeros_like(self.adj_matrix, dtype=int) for _ in range(self.k)]

        current_layer = set(initial_sample)

        cnt = len(current_layer)

        for i in range(self.k):
            next_layer = set()
            for node in current_layer:

                nonzero_indices = self.adj_matrix[node].nonzero()
                neighbors = set(nonzero_indices.reshape(-1).tolist())

                sampled_neighbors = random.sample(neighbors, min(len(neighbors), self.disk_read_nodes_per_hop))
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


    def FBL_loader(self):

        return self.batch_base_nodes, self.compute_matrices
