import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSDSampler:
    def __init__(self, data, N):
        self.data = data
        self.static_sampled_nodes = self.preprocess_static_sampling(N)

    def preprocess_static_sampling(self, N):
        return random.sample(list(self.data), N)

    def resample(self, n, alpha):
        non_static_nodes = list(set(list(self.data)) - set(self.static_sampled_nodes))
        dynamic_resampled_nodes = random.sample(non_static_nodes, int((1-alpha)*n))
        static_resampled_nodes = random.sample(self.static_sampled_nodes, int(alpha*n))
        combined_resampled_nodes = dynamic_resampled_nodes + static_resampled_nodes
        cutoff_dynamic_resampled_nodes = random.sample(static_resampled_nodes, int(alpha*n))

        dynamic_resampled_nodes_swap_in = random.sample(non_static_nodes, int(alpha*n))

        for node in cutoff_dynamic_resampled_nodes:
            self.static_sampled_nodes.remove(node)
        self.static_sampled_nodes.extend(dynamic_resampled_nodes_swap_in)

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
