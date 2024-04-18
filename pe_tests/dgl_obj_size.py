import torch
from ogb.nodeproppred import DglNodePropPredDataset
import sys

def get_memory_usage_dgl(dataset_name):
    dataset = DglNodePropPredDataset(name=dataset_name)
    graph, label = dataset[0]  # 加载图和标签
    
    # 计算图的内存占用
    graph_size = sum(tensor.element_size() * tensor.nelement() for tensor in graph.ndata.values())
    graph_size += sum(tensor.element_size() * tensor.nelement() for tensor in graph.edata.values())

    # 标签的内存占用
    if isinstance(label, torch.Tensor):
        label_size = label.element_size() * label.nelement()
    else:
        label_size = sys.getsizeof(label)  # 如果label不是张量，使用sys.getsizeof

    total_size = graph_size + label_size
    return total_size

# 数据集名称
datasets = ["ogbn-products", "ogbn-arxiv", "ogbn-mag"]

# 打印每个数据集的内存大小
for ds in datasets:
    size = get_memory_usage_dgl(ds)
    print(f"The estimated memory usage of {ds} is {size} bytes.")
