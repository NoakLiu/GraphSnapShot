import matplotlib.pyplot as plt
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset

# 要分析的数据集列表
dataset_names = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']

for dataset_name in dataset_names:
    # 加载数据集
    dataset = DglNodePropPredDataset(name=dataset_name)
    graph, _ = dataset[0]  # 注意这里使用了元组解包，将图和标签分开

    # 获取节点的度
    deg = graph.in_degrees() + graph.out_degrees()
    print(deg)

    # 统计度分布
    deg_hist = torch.bincount(deg)

    # 绘制度分布图
    plt.loglog(deg_hist.numpy(), 'bo-')
    plt.title('Degree Distribution for ' + dataset_name)
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')

    # 保存图形
    plt.savefig(dataset_name + '_degree_distribution.png')

    # 显示图形
    plt.show()

# Load dataset
dataset_name = 'ogbn-mag'
dataset = DglNodePropPredDataset(name=dataset_name)
graph, _ = dataset[0]

# Inspect available edge types
print("Available edge types:", graph.canonical_etypes)

# Initialize degree tensor
deg = torch.zeros(graph.number_of_nodes())

# Calculate node degrees for each edge type and accumulate
for etype in graph.canonical_etypes:
    in_deg = graph.in_degrees(etype=etype)
    out_deg = graph.out_degrees(etype=etype)
    deg += in_deg + out_deg

# Count degree distribution
deg_hist = torch.bincount(deg.int())

# Plot degree distribution
plt.loglog(deg_hist.numpy(), 'bo-')
plt.title('Degree Distribution for ' + dataset_name)
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')

# Save plot
plt.savefig(dataset_name + '_degree_distribution.png')

# Display plot
plt.show()
