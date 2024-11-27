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

    # def forward(self, input, adj_list):
    #     # Aggregate features from each hop
    #     aggregated_features = []
    #     for adj in adj_list:
    #         h = torch.mm(input, self.W)
    #         if self.model_mode == "dense":
    #             support = torch.mm(adj, h)
    #         # elif self.model_mode == "sparse":
    #         #     support = torch.spmm(adj.long(), h)
    #         elif self.model_mode == "sparse":
    #             print(adj.dtype)
    #             print(h.dtype)
    #             support = torch.spmm(adj, h)  # 注意这里
    #             # adj_sparse = adj.to_sparse()  # 转换为稀疏张量
    #             # print("adj_sparse:\n",adj_sparse)
    #             # print("h:\n",h)
    #             # support = torch.spmm(adj_sparse.coalesce(), h)
    #
    #         aggregated_features.append(support)
    #
    #     # Sum features from all hops
    #     h = sum(aggregated_features)
    #     return F.relu(h)

    def forward(self, input, adj_list):
        aggregated_features = []
        k = 0
        for adj in adj_list:
            # print("k=",k)
            k+=1
            h = torch.mm(input, self.W)

            print(adj.shape)
            print(adj.dtype)

            """
            torch.Size([3327, 500])
            torch.Size([500, 500])
            torch.float32
            torch.Size([500, 500])
            torch.float32
            torch.Size([500, 500])
            torch.float32
            torch.Size([500, 500])
            torch.float32
            """
            if self.model_mode == "sparse":
                # Convert dense to sparse if necessary
                if not adj.is_sparse:
                    adj = adj.to_sparse()

                # Ensure the indices are of type Long
                adj_indices = adj.indices().type(torch.LongTensor)
                adj_values = adj.values()
                adj_shape = adj.shape
                adj_sparse = torch.FloatTensor(adj_indices, adj_values, adj_shape).to_sparse()

                support = torch.spmm(adj_sparse, h)
            else:
                # print(adj.dtype)
                # print(h.dtype)
                # print("adj shape:",adj.shape)
                # print("h.shape",h.shape)
                support = torch.mm(adj, h)

            aggregated_features.append(support)

        h = sum(aggregated_features)
        return F.relu(h)
class kHopGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, model_mode="dense"):
        super(kHopGCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(nfeat, nhid, dropout, model_mode=model_mode)
        self.gc2 = GraphConvolutionLayer(nhid, nclass, dropout, model_mode=model_mode)
        self.dropout = dropout

    def forward(self, x, adj_list):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, adj_list)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_list)
        return F.log_softmax(x, dim=1)