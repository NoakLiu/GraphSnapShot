import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Disk_Mem_Simulation.simulation_benchmark import benchmark

class GraphSDSampler:
    def __init__(self, data, N):
        self.data = data
        self.static_sampled_nodes = self.preprocess_static_sampling(N)

        self.k1 = 0
        self.k2 = 0
        self.k3 = 0

    def preprocess_static_sampling(self, N):
        return random.sample(list(self.data), N)

    def resample(self, n, alpha, mode = "no exchange"):

        non_static_nodes = list(set(list(self.data)) - set(self.static_sampled_nodes))

        self.k1+=int((1-alpha)*n)
        if(self.k1 % 10)==0:
            benchmark.simulate_disk_read()
            benchmark.simulate_memory_access()
        dynamic_resampled_nodes = random.sample(non_static_nodes, int((1-alpha)*n))

        self.k2+=int(alpha*n)
        if(self.k2 % 10)==0:
            benchmark.simulate_memory_access()
            benchmark.simulate_memory_access()

        static_resampled_nodes = random.sample(self.static_sampled_nodes, int(alpha*n))

        combined_resampled_nodes = dynamic_resampled_nodes + static_resampled_nodes

        if(mode=="no exchange"):

            cutoff_dynamic_resampled_nodes = random.sample(static_resampled_nodes, int(alpha * n))

            dynamic_resampled_nodes_swap_in = random.sample(non_static_nodes, int(alpha*n))

            for node in cutoff_dynamic_resampled_nodes:
                self.static_sampled_nodes.remove(node)
            self.static_sampled_nodes.extend(dynamic_resampled_nodes_swap_in)

            self.k3 += int(alpha * n)
            if (self.k3 % 10) == 0:
                benchmark.simulate_disk_read()
                benchmark.simulate_disk_write()
                benchmark.simulate_memory_access()
        else:
            cutoff_dynamic_resampled_nodes = random.sample(static_resampled_nodes, int(min(alpha, 1-alpha)* n))

            dynamic_resampled_nodes_swap_in = random.sample(dynamic_resampled_nodes, int(min(alpha, 1-alpha)* n))

            for node in cutoff_dynamic_resampled_nodes:
                self.static_sampled_nodes.remove(node)
            self.static_sampled_nodes.extend(dynamic_resampled_nodes_swap_in)

            self.k3 += int(min(alpha, 1-alpha)* n)
            if (self.k3 % 10) == 0:
                benchmark.simulate_memory_access()
                benchmark.simulate_disk_write()
                benchmark.simulate_memory_access()

        return combined_resampled_nodes


# Native MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, features, sampled_edges):
        h = self.fc1(features)
        h = torch.spmm(sampled_edges, h)
        h = F.relu(h)
        h = self.fc2(h)
        return F.log_softmax(h, dim=1)
