import dgl
import psutil
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    NeighborSampler_FCR_struct_shared_cache,
    NeighborSampler_FCR_struct,
    NeighborSampler_OTF_struct_shared_cache,
    NeighborSampler_OTF_struct,
    MultiLayerNeighborSampler,
    BlockSampler
)

# 创建一个DGL图
# g = dgl.graph(([0, 1, 2], [1, 2, 3]))

dataset = DglNodePropPredDataset("ogbn-products")
dataset = AsNodePredDataset(dataset)

g = dataset[0]

# 获取DGL图的内存使用情况
memory_usage = psutil.Process().memory_info().rss
print("Memory Usage (before creating graph):", memory_usage / (1024 * 1024), "MB")

# 强制触发图的构建，以便占用内存
g.create_formats_()

# 获取DGL图的内存使用情况
memory_usage = psutil.Process().memory_info().rss
print("Memory Usage (after creating graph):", memory_usage / (1024 * 1024), "MB")
