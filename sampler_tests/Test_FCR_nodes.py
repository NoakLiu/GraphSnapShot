import dgl
from dgl.dataloading import NeighborSampler
import torch

import argparse
import time

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    MultiLayerNeighborSampler,
    BlockSampler
)

class NeighborSampler_FCR(NeighborSampler):
    def __init__(self, g, alpha, beta, k, fanouts, **kwargs):
        """
        Initialize the NeighborSampler_FCR.

        Parameters:
        g (DGLGraph): The input graph.
        alpha (float): The fraction of nodes to include in g_static.
        beta (float): The fraction of g_static nodes to replace every k epochs.
        k (int): The number of epochs after which g_static is partially updated.
        fanouts (list[int] or list[dict[etype, int]]): List of neighbors to sample per edge type for each GNN layer.
        **kwargs: Additional arguments for the NeighborSampler.
        """
        super().__init__(fanouts, **kwargs)
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.epoch_counter = 0

        # Preprocess to split the graph
        self.preprocess()

    def preprocess(self):
        """
        Splits the input graph into g_static and g_dynamic based on alpha.
        """
        num_nodes = self.g.number_of_nodes()
        num_static_nodes = int(num_nodes * self.alpha)

        # Randomly sample nodes for g_static
        static_nodes = torch.randperm(num_nodes)[:num_static_nodes]

        # Create g_static and g_dynamic
        g_static = self.g.subgraph(static_nodes).to(device)
        dynamic_nodes = torch.tensor(list(set(range(num_nodes)) - set(static_nodes.tolist())))
        g_dynamic = self.g.subgraph(dynamic_nodes)

        self.g_static, self.g_dynamic = g_static, g_dynamic

    def cache_refresh(self):
        """
        Updates g_static by discarding beta fraction of its nodes and replacing
        them with the same number of nodes from g_dynamic through disk_cache_swap.
        """
        num_static_nodes = self.g_static.number_of_nodes()
        num_nodes_to_replace = int(num_static_nodes * self.beta)

        # Nodes to discard from g_static and to add from g_dynamic
        nodes_to_discard = torch.randperm(num_static_nodes)[:num_nodes_to_replace]
        nodes_to_add = torch.randperm(self.g_dynamic.number_of_nodes())[:num_nodes_to_replace]

        # Perform the swap
        self.disk_cache_swap(nodes_to_discard, nodes_to_add)

    def disk_cache_swap(self, nodes_to_discard, nodes_to_add):
        """
        Performs the actual swapping of nodes between g_static and g_dynamic.
        """
        # Update g_static
        remaining_static_nodes = torch.tensor(list(set(range(self.g_static.number_of_nodes())) - set(nodes_to_discard.tolist())))
        new_static_nodes = torch.cat([self.g_static.ndata[dgl.NID][remaining_static_nodes], self.g_dynamic.ndata[dgl.NID][nodes_to_add]])
        self.g_static = self.g.subgraph(new_static_nodes).to(device)

        # Update g_dynamic
        remaining_dynamic_nodes = torch.tensor(list(set(range(self.g_dynamic.number_of_nodes())) - set(nodes_to_add.tolist())))
        new_dynamic_nodes = torch.cat([self.g_dynamic.ndata[dgl.NID][remaining_dynamic_nodes], self.g_static.ndata[dgl.NID][nodes_to_discard]])
        self.g_dynamic = self.g.subgraph(new_dynamic_nodes)

    def sample_blocks_FCR(self, seed_nodes, exclude_eids=None):
        """
        Overrides the sample_blocks method to incorporate the logic of updating g_static
        every k epochs.
        """
        if self.epoch_counter % self.k == 0:
            self.cache_refresh()

        self.epoch_counter += 1

        # Call the original sample_blocks method using g_static
        return super().sample_blocks(self.g_static, seed_nodes, exclude_eids)
    

# NeighborSampler_FCR(g,0.1,0.2,3,[2,2,2])
    
from ogb.nodeproppred import DglNodePropPredDataset

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, hidden_size))
        num_layers = 3
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GraphConv(hidden_size, hidden_size))
        self.layers.append(dglnn.GraphConv(hidden_size, out_feats))

        self.hidden_size = hidden_size
        self.out_size = out_feats
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        for layer, block in zip(self.layers, blocks):
            x = layer(block, x)
            x = F.relu(x)
        return x

    def inference(self, g, device, batch_size, fused_sampling: bool = True):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        
        sampler = MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["feat"], fused=fused_sampling
        )

        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        # Enable pin_memory for faster CPU to GPU data transfer if the
        # model is running on a GPU.
        pin_memory = buffer_device != device

        print(self.layers)

        """
        ModuleList(
        (0): GraphConv(in=100, out=256, normalization=both, activation=None)
        (1): GraphConv(in=256, out=256, normalization=both, activation=None)
        (2): GraphConv(in=256, out=47, normalization=both, activation=None)
        )
        """

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1
            y = torch.empty(
                g.num_nodes(),
                self.out_size if is_last_layer else self.hidden_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                # print(input_nodes, output_nodes, blocks)
                x = feat[input_nodes]
                # hidden_x = layer(blocks[0], x)  # len(blocks) = 1

                hidden_x = x

                for block in blocks:
                    # Perform message passing on the current block
                    hidden_x = layer(block, hidden_x)

                if layer_idx != len(self.layers) - 1:
                    hidden_x = F.relu(hidden_x)
                    hidden_x = self.dropout(hidden_x)
                # By design, our output nodes are contiguous.
                y[output_nodes[0] : output_nodes[-1] + 1] = hidden_x.to(
                    buffer_device
                )
            feat = y
        return y


@torch.no_grad()
def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        x = blocks[0].srcdata["feat"]
        ys.append(blocks[-1].dstdata["label"])
        y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


@torch.no_grad()
def layerwise_infer(
    device, graph, nid, model, num_classes, batch_size, fused_sampling
):
    model.eval()
    pred = model.inference(
        graph, device, batch_size, fused_sampling
    )  # pred in buffer_device.
    pred = pred[nid]
    label = graph.ndata["label"][nid].to(pred.device)
    return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)


def train(device, g, dataset, model, num_classes, use_uva, fused_sampling):
    # Create sampler & dataloader.
    train_idx = dataset.train_idx.to(g.device if not use_uva else device)
    val_idx = dataset.val_idx.to(g.device if not use_uva else device)
    sampler_cuda = NeighborSampler_FCR(
        g=g,
        fanouts=[2, 2, 2],  # fanout for [layer-0, layer-1, layer-2]
        alpha=0.2,
        beta=0.3,
        k=3,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
        fused=fused_sampling,
    )

    sampler = NeighborSampler_FCR(
        g=g,
        fanouts=[4, 4, 4],  # fanout for [layer-0, layer-1, layer-2]
        alpha=0.2,
        beta=0.3,
        k=3,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
        fused=fused_sampling,
    )

    # 调用示例
    seed_nodes1, output_nodes1, blocks1 = sampler.sample_blocks_FCR(2)
    print("1sampler")
    print(seed_nodes1)
    print(output_nodes1)
    print(blocks1[0])
    print(blocks1[0].edata)
    print("---")
    seed_nodes2, output_nodes2, blocks2 = sampler_cuda.sample_blocks_FCR(3)
    print("2sampler")
    print(seed_nodes2)
    print(output_nodes2)
    print(blocks2)
    print("---")

    # merged_seed_nodes, merged_output_nodes, merged_blocks = merge_neighbor_samplers(sampler, sampler_cuda, seed_nodes1, seed_nodes2, output_nodes1, output_nodes2, blocks1, blocks2)

    # print("result")
    # print(merged_seed_nodes)
    # print(merged_output_nodes)
    # print(merged_blocks)
    # print("---")

    # sampler_res = merge_neighbor_samplers(sampler, sampler_cuda)

    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        # If `g` is on gpu or `use_uva` is True, `num_workers` must be zero,
        # otherwise it will cause error.
        num_workers=0,
        use_uva=use_uva,
    )

    train_dataloader_cuda = DataLoader(
        g,
        train_idx,
        sampler_cuda,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        # If `g` is on gpu or `use_uva` is True, `num_workers` must be zero,
        # otherwise it will cause error.
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        # No need to shuffle for validation.
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader_cuda = DataLoader(
        g,
        val_idx,
        sampler_cuda,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        # If `g` is on gpu or `use_uva` is True, `num_workers` must be zero,
        # otherwise it will cause error.
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(1):
        t0 = time.time()
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # The input features from the source nodes in the first layer's
            # computation graph.
            x = blocks[0].srcdata["feat"]

            # The ground truth labels from the destination nodes
            # in the last layer's computation graph.
            y = blocks[-1].dstdata["label"]

            print(blocks)

            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        t1 = time.time()
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | "
            f"Accuracy {acc.item():.4f} | Time {t1 - t0:.4f}"
        )

    print("middle")

    for epoch in range(1):
        t0 = time.time()
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader_cuda
        ):
            # The input features from the source nodes in the first layer's
            # computation graph.

            print(blocks)

            x = blocks[0].srcdata["feat"]

            # The ground truth labels from the destination nodes
            # in the last layer's computation graph.
            y = blocks[-1].dstdata["label"]

            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        t1 = time.time()
        acc = evaluate(model, g, val_dataloader_cuda, num_classes)
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | "
            f"Accuracy {acc.item():.4f} | Time {t1 - t0:.4f}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "gpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for "
        "CPU-GPU mixed training, 'gpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--compare-to-graphbolt",
        default="false",
        choices=["false", "true"],
        help="Whether comparing to GraphBolt or not, 'false' by default.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # Load and preprocess dataset.
    print("Loading data")
    # dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))

    dataset = DglNodePropPredDataset("ogbn-products")
    dataset = AsNodePredDataset(dataset)

    g = dataset[0]


    # Add self-loops to the graph
    g = dgl.add_self_loop(g)


    if args.compare_to_graphbolt == "false":
        g = g.to("cuda" if args.mode == "gpu" else "cpu")
    num_classes = dataset.num_classes
    # Whether use Unified Virtual Addressing (UVA) for CUDA computation.
    use_uva = args.mode == "mixed"
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    fused_sampling = args.compare_to_graphbolt == "false"

    # Create GCN model.
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = GCN(in_size, 256, out_size).to(device)

    # Model training.
    print("Training...")
    train(device, g, dataset, model, num_classes, use_uva, fused_sampling)

    # Test the model.
    print("Testing...")
    acc = layerwise_infer(
        device,
        g,
        dataset.test_idx,
        model,
        num_classes,
        batch_size=4096,
        fused_sampling=fused_sampling,
    )
    print(f"Test accuracy {acc.item():.4f}")
