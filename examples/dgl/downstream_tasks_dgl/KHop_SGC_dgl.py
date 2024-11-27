import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import CoraGraphDataset
from dgl.nn import GraphConv

# 加载 DGL 的 CORA 数据集
dataset = CoraGraphDataset()
g = dataset[0]
g = dgl.add_self_loop(g)  # 为图添加自环

# 定义 SGC 模型
class SGC(nn.Module):
    def __init__(self, in_feats, out_feats, k):
        super(SGC, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, allow_zero_in_degree=True)
        self.k = k

    def forward(self, g, features):
        with g.local_scope():
            g.ndata['h'] = features
            for _ in range(self.k):
                # 聚合邻居的特征
                g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            h = g.ndata['h']
            h = self.conv(g, h)
            return h

# 设置超参数
in_feats = g.ndata['feat'].shape[1]
out_feats = dataset.num_classes
k = 2  # SGC中的跳数

# 初始化模型和优化器
model = SGC(in_feats, out_feats, k)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练模型
def train(model, g, epochs):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']

    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(1)[val_mask] == labels[val_mask]).float().mean()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Acc: {acc.item()}')

train(model, g, epochs=200)

# 评估模型
def evaluate(model, g):
    features = g.ndata['feat']
    labels = g.ndata['label']
    test_mask = g.ndata['test_mask']

    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        acc = (logits.argmax(1)[test_mask] == labels[test_mask]).float().mean()
        print(f'Test Acc: {acc.item()}')

evaluate(model, g)
