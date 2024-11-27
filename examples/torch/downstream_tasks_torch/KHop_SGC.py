import torch
import torch.nn as nn
class kHopSGC(nn.Module):
    def __init__(self, in_features, out_features, k, model_mode="sparse"):
        """
        :param in_features: Number of input features
        :param out_features: Number of output features
        :param adj_list: List of adjacency matrices for 1-hop to K-hop
        """
        super(kHopSGC, self).__init__()
        # self.adj_list = adj_list
        self.linear = nn.Linear(in_features * k, out_features)
        self.model_mode = model_mode

    def forward(self, x, adj_list):
        """
        :param x: Node feature matrix
        """
        # Aggregate features from each hop
        aggregated_features = []

        for adj in adj_list:
            # h = torch.mm(input, self.W)

            print(adj.shape)
            print(adj.dtype)

            """
            torch.Size([30, 30])
            torch.float32
            """

            if self.model_mode == "sparse":
                # Convert dense to sparse if necessary
                # if not adj.is_sparse:
                #     adj = adj.to_sparse()
                #
                # # Ensure the indices are of type Long
                # adj_indices = adj.indices().type(torch.LongTensor)
                # adj_values = adj.values()
                # adj_shape = adj.shape
                # # adj_sparse = torch.FloatTensor(adj_indices, adj_values, adj_shape).to_sparse(sparseDims=2)
                # adj_sparse = torch.FloatTensor(adj_indices, adj_values, adj_shape).to_sparse()
                # # adj_sparse = torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)

                if not adj.is_sparse:
                    adj = adj.to_sparse()

                # Ensure the indices are of type Long
                adj_indices = adj.indices().type(torch.LongTensor)
                adj_values = adj.values()
                adj_shape = adj.shape
                adj_sparse = torch.FloatTensor(adj_indices, adj_values, adj_shape).to_sparse()

                support = torch.spmm(adj_sparse.long(), x)
            else:
                # print(adj.dtype)
                # print(h.dtype)
                # print("adj shape:",adj.shape)
                # print("h.shape",h.shape)
                support = torch.mm(adj, x)

            aggregated_features.append(support)

        # for adj in adj_list:
        #     aggregated_features.append(torch.spmm(adj.long(), x))

        # Concatenate features from all hops
        x = torch.cat(aggregated_features, dim=1)
        x = self.linear(x)
        return x