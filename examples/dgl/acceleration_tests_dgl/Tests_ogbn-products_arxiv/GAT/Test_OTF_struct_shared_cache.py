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
    BlockSampler,
    NeighborSampler_OTF_struct_shared_cache
)
from ogb.nodeproppred import DglNodePropPredDataset

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats,n_layers, num_heads):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.out_size = out_feats
        self.num_heads = num_heads
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(
                (in_feats, in_feats),
                hidden_size,
                num_heads=num_heads,
                activation=F.relu,
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                dglnn.GATConv(
                    (hidden_size * num_heads, hidden_size * num_heads),
                    hidden_size,
                    num_heads=num_heads,
                    activation=F.relu,
                )
            )
        self.layers.append(
            dglnn.GATConv(
                (hidden_size * num_heads, hidden_size * num_heads),
                out_feats,
                num_heads=num_heads,
                activation=None,
            )
        )
    
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h.log_softmax(dim=-1)
    
    def inference(self, g, device, batch_size, fused_sampling: bool = True):
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

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1
            # y = torch.empty(
            #     g.num_nodes(),
            #     self.out_size if is_last_layer else self.hidden_size,
            #     device=buffer_device,
            #     pin_memory=pin_memory,
            # )

            if layer_idx < self.n_layers - 1:
                y = torch.zeros(
                    g.num_nodes(),
                    self.hidden_size * self.num_heads
                    if layer_idx != len(self.layers) - 1
                    else self.out_size,
                )
            else:
                y = torch.zeros(
                    g.num_nodes(),
                    self.hidden_size
                    if layer_idx != len(self.layers) - 1
                    else self.out_size,
                )

            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):

                block = blocks[0].int().to(device)

                h = feat[input_nodes].to(device)
                h_dst = h[: block.num_dst_nodes()]
                if layer_idx < self.n_layers - 1:
                    h = layer(block, (h, h_dst)).flatten(1)
                else:
                    h = layer(block, (h, h_dst))
                    h = h.mean(1)
                    h = h.log_softmax(dim=-1)

                y[output_nodes] = h.cpu()
            feat = y
        return y

@torch.no_grad()
def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = blocks[::-1]
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

    sampler_cuda = NeighborSampler_OTF_struct_shared_cache(
        g=g,
        fanouts=[2,2,2],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
        alpha=2, beta=2, gamma=0.15, T=119,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
        fused=fused_sampling,
    )

    sampler = NeighborSampler_OTF_struct_shared_cache(
        g=g,
        fanouts=[4,4,4],  # fanout for [layer-0, layer-1, layer-2] [4,4,4]
        alpha=2, beta=2, gamma=0.15, T=119,#3, 0.4
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
        fused=fused_sampling,
    )

    # 调用示例
    seed_nodes1, output_nodes1, blocks1 = sampler.sample_blocks(g, 2)
    print("1sampler")
    print(seed_nodes1)
    print(output_nodes1)
    print(blocks1[0])
    print(blocks1[0].edata)
    print("---")
    seed_nodes2, output_nodes2, blocks2 = sampler_cuda.sample_blocks(g, 3)
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
            blocks = blocks[::-1]
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
            blocks = blocks[::-1]
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
    # model = GCN(in_size, 256, out_size).to(device)
    model = GAT(in_feats=in_size,hidden_size=256,out_feats=out_size,n_layers=3,num_heads=8)

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