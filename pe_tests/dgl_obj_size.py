import torch
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
import sys
import psutil
from pympler import asizeof

def get_memory_usage_dgl(dataset_name):
    size_pre =  psutil.virtual_memory().used
    dataset = DglNodePropPredDataset(name=dataset_name)
    dataset = AsNodePredDataset(dataset)
    graph = dataset[0]  # 加载图和标签
    size_lat =  psutil.virtual_memory().used
    # total_size = (size_lat - size_pre)/(1024**2)

    # total_size = sys.getsizeof(graph)
    
    total_size = asizeof.asizeof(graph)

    g_sample = graph.sample_neighbors(
        torch.arange(0, graph.number_of_nodes()), 5
    )
    gs_size = asizeof.asizeof(g_sample)

    return total_size, gs_size

# 数据集名称
datasets = ["ogbn-products", "ogbn-arxiv",]

# 打印每个数据集的内存大小
for ds in datasets:
    size, gs_size= get_memory_usage_dgl(ds)
    print(f"The estimated memory usage of {ds} is {size} bytes.")
    print(f"The estimate memory usage of sampled {ds} is {gs_size} bytes.")
