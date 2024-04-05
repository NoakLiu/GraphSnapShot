import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from Disk_Mem_Simulation.simulation_benchmark import benchmark


class GraphSDSampler:
    def __init__(self, data, N, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.static_sampled_nodes = [self.preprocess_static_sampling(N) for _ in range(batch_size)]

        self.k1 = [0] * batch_size
        self.k2 = [0] * batch_size
        self.k3 = [0] * batch_size

    def preprocess_static_sampling(self, N):
        return random.sample(list(self.data), N)

    def resample(self, n, alpha, mode="no exchange"):
        batch_combined_resampled_nodes = []

        for i in range(self.batch_size):
            non_static_nodes = list(set(self.data) - set(self.static_sampled_nodes[i]))

            self.k1[i] += int((1 - alpha) * n)
            if self.k1[i] % 10 == 0:
                benchmark.simulate_disk_read()
                benchmark.simulate_memory_access()

            dynamic_resampled_nodes = random.sample(non_static_nodes, int((1 - alpha) * n))

            self.k2[i] += int(alpha * n)
            if self.k2[i] % 10 == 0:
                benchmark.simulate_memory_access()
                benchmark.simulate_memory_access()

            static_resampled_nodes = random.sample(self.static_sampled_nodes[i], int(alpha * n))
            combined_resampled_nodes = dynamic_resampled_nodes + static_resampled_nodes

            if mode == "no exchange":
                self.handle_no_exchange(alpha, n, dynamic_resampled_nodes, static_resampled_nodes, i)
            else:
                self.handle_exchange(alpha, n, dynamic_resampled_nodes, static_resampled_nodes, i)

            batch_combined_resampled_nodes.append(combined_resampled_nodes)

        return batch_combined_resampled_nodes

    def handle_no_exchange(self, alpha, n, dynamic_nodes, static_nodes, batch_index):
        cutoff_nodes = random.sample(static_nodes, int(alpha * n))
        swap_in_nodes = random.sample(dynamic_nodes, int(alpha * n))
        self.update_static_sampled_nodes(cutoff_nodes, swap_in_nodes, batch_index)
        self.k3[batch_index] += int(alpha * n)
        self.simulate_benchmark(self.k3[batch_index])

    def handle_exchange(self, alpha, n, dynamic_nodes, static_nodes, batch_index):
        cutoff_nodes = random.sample(static_nodes, int(min(alpha, 1 - alpha) * n))
        swap_in_nodes = random.sample(dynamic_nodes, int(min(alpha, 1 - alpha) * n))
        self.update_static_sampled_nodes(cutoff_nodes, swap_in_nodes, batch_index)
        self.k3[batch_index] += int(min(alpha, 1 - alpha) * n)
        self.simulate_benchmark(self.k3[batch_index])

    def update_static_sampled_nodes(self, cutoff_nodes, swap_in_nodes, batch_index):
        for node in cutoff_nodes:
            self.static_sampled_nodes[batch_index].remove(node)
        self.static_sampled_nodes[batch_index].extend(swap_in_nodes)

    def simulate_benchmark(self, k):
        if k % 10 == 0:
            benchmark.simulate_disk_read()
            benchmark.simulate_disk_write()
            benchmark.simulate_memory_access()


# MLP model modified for batch processing
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, features, sampled_edges):
        h = F.relu(self.fc1(features))
        batch_h = torch.stack([torch.spmm(edge, h) for edge in sampled_edges])
        batch_h = F.relu(batch_h)
        batch_output = self.fc2(batch_h)
        return F.log_softmax(batch_output, dim=1)

# Example usage:
# sampler = GraphSDSampler(data, N=100, batch_size=10)
# model = MLP(input_dim, output_dim)
