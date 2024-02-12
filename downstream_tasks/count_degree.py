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