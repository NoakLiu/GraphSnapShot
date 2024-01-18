import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, model_mode="sparse"):
        super(GraphConvolutionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.model_mode = model_mode  # "dense" or "sparse"

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, input, adj_list):
        # Aggregate features from each hop
        aggregated_features = []
        for adj in adj_list:
            h = torch.mm(input, self.W)
            if self.model_mode == "dense":
                support = torch.mm(adj, h)
            elif self.model_mode == "sparse":
                support = torch.spmm(adj, h)
            aggregated_features.append(support)

        # Sum features from all hops
        h = sum(aggregated_features)
        return F.relu(h)

class kHopGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K, adj_list, model_mode="sparse"):
        super(kHopGCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(nfeat, nhid, dropout, model_mode=model_mode)
        self.gc2 = GraphConvolutionLayer(nhid, nclass, dropout, model_mode=model_mode)
        self.dropout = dropout

        # Compute and store the adjacency list for K hops
        self.adj_list = adj_list

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, self.adj_list)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj_list)
        return F.log_softmax(x, dim=1)