import torch
import torch.nn as nn
class KHopSGC(nn.Module):
    def __init__(self, in_features, out_features, adj_list):
        """
        :param in_features: Number of input features
        :param out_features: Number of output features
        :param adj_list: List of adjacency matrices for 1-hop to K-hop
        """
        super(KHopSGC, self).__init__()
        self.adj_list = adj_list
        self.linear = nn.Linear(in_features * len(adj_list), out_features)

    def forward(self, x):
        """
        :param x: Node feature matrix
        """
        # Aggregate features from each hop
        aggregated_features = []
        for adj in self.adj_list:
            aggregated_features.append(torch.spmm(adj, x))
        # Concatenate features from all hops
        x = torch.cat(aggregated_features, dim=1)
        x = self.linear(x)
        return x