from sympy import degree, im
import torch
from ogb.nodeproppred import NodePropPredDataset
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# 下载并加载 OGB-MAG 数据集
dataset = NodePropPredDataset(name="ogbn-mag")

# 提取边信息
# edge_index = dataset[0].edge_index
# edge_index = dataset[0]['edge_index_dict']

# Assuming edge information is the second element (check library documentation)
edge_index = dataset[0][0]  # Access the second element (might differ based on library)
print(edge_index)

# # Separate source and destination nodes (assuming edge_index is a list of lists)
# source_nodes = [edge[0] for edge in edge_index]
# destination_nodes = [edge[1] for edge in edge_index]

# Compute node degrees
edge_index_dict = edge_index['edge_index_dict']
edge_types = [('author', 'affiliated_with', 'institution'), ('author', 'writes', 'paper')]
edges = np.concatenate([edge_index_dict[edge_type] for edge_type in edge_types], axis=1)
print(edges)

# Flatten the array to get a single list of all involved nodes
all_nodes = np.concatenate((edges[0], edges[1]))

# Compute unique nodes and their degree (counts)
unique_nodes, degrees = np.unique(all_nodes, return_counts=True)

# Plotting the degree distribution
plt.figure(figsize=(10, 6))
# Using a log-log plot for better visualization of wide-range distributions
plt.loglog(range(len(degrees)), np.sort(degrees)[::-1], marker='o', linestyle='None')
plt.title('Degree Distribution')
plt.xlabel('Rank')
plt.ylabel('Degree')
plt.show()

# Alternatively, you can plot a histogram of the degrees
plt.figure(figsize=(10, 6))
deg_hist, bins = np.histogram(degrees, bins=50)
plt.loglog(bins[:-1], deg_hist, 'bo-')
plt.title('Degree Distribution Histogram')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.show()

# unique_nodes, node_counts = np.unique(edges[0], return_counts=True)
# node_degrees = dict(zip(unique_nodes, node_counts))

# # Extract node degrees
# degrees = torch.tensor(list(node_degrees.values()))

# # print(degrees)

# # # # Plot histogram
# # # plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
# # # plt.title('Degree Distribution')
# # # plt.xlabel('Degree')
# # # plt.ylabel('Frequency')
# # # plt.show()

# # deg_hist, bins = np.histogram(degrees, bins=50)  # Adjust the number of bins as needed

# # # Plot degree distribution
# # plt.loglog(bins[:-1], deg_hist, 'bo-')
# # plt.title('Degree Distribution')
# # plt.xlabel('Degree')
# # plt.ylabel('Number of Nodes')
# # plt.show()

# deg_hist = torch.bincount(degrees)

# # 绘制度分布图
# plt.loglog(deg_hist.numpy(), 'bo-')
# plt.title('Degree Distribution for ' + "ogbn-mag")
# plt.xlabel('Degree')
# plt.ylabel('Number of Nodes')

# # 保存图形
# plt.savefig("ogbn-mag" + '_degree_distribution.png')

# # 显示图形
# plt.show()





# # # 统计每个节点的度
# # degree_counts = Counter(edge_index.view(-1).tolist())

# # # 提取度及其对应数量
# # degrees = list(degree_counts.keys())
# # counts = list(degree_counts.values())

# # # 绘制度分布图
# # plt.figure(figsize=(8, 6))
# # plt.bar(degrees, counts)
# # plt.xlabel("节点度")
# # plt.ylabel("频率")
# # plt.title("OGBN-MAG 中的度分布")
# # plt.yscale("log")  # 使用对数刻度以更好地显示大范围变化

# # plt.show()
