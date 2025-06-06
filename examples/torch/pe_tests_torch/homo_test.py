import psutil
import os

# Before the operation
process = psutil.Process(os.getpid())


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
    NeighborSampler_FCR_struct,
    # NeighborSampler_FCR_struct_shared_cache,
    # NeighborSampler_OTF_struct,
    # NeighborSampler_OTF_struct_shared_cache

)
from ogb.nodeproppred import DglNodePropPredDataset

def train(device, g, dataset, num_classes, use_uva, fused_sampling, mem_before):
    # Create sampler & dataloader.
    train_idx = dataset.train_idx.to(g.device if not use_uva else device)
    val_idx = dataset.val_idx.to(g.device if not use_uva else device)

    # FBL
    sampler = NeighborSampler(
        [20, 20, 20],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
        fused=fused_sampling,
    )

    # FCR
    # sampler = NeighborSampler_FCR_struct(
    #     g=g,
    #     fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     alpha=2, T=50,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # FCR shared cache
    # sampler = NeighborSampler_FCR_struct_shared_cache(
    #     g=g,
    #     fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     alpha=2, T=50,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )    

    # OTF shared cache
    # sampler = NeighborSampler_OTF_struct(
    #     g=g,
    #     fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [4,4,4]
    #     alpha=2, beta=1, gamma=0.15, T=358, #3, 0.4
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # # OTF shared cache
    # sampler = NeighborSampler_OTF_struct_shared_cache(
    #     g=g,
    #     fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     alpha=2, beta=1, gamma=0.15, T=119,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    seed_nodes1, output_nodes1, blocks1 = sampler.sample_blocks(g, 2)
    print("1sampler")
    print(seed_nodes1)
    print(output_nodes1)
    print(blocks1[0])
    print(blocks1[0].edata)
    print("---")
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
    lstime = []
    lsmem = []

    for epoch in range(3):
        t0 = time.time()
        t20 = time.time()
        total_loss = 0
        # Before the operation
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # After the operation
            mem_after = psutil.virtual_memory().used
            memusage = process.memory_info().rss / (1024 ** 3) 
            t2=time.time()
            lsmem.append(memusage)
            lstime.append(t2-t20)
            print("memorage usage (GB):",memusage)
            print("time needed (s):",t2-t20)
            # Before the operation
            # mem_before = psutil.virtual_memory().used
            t20=time.time()
        t1 = time.time()
        print("lstime.mean:",sum(lstime)/len(lstime))
        print("lsmem.mean",sum(lsmem)/len(lsmem))
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | Time {t1 - t0:.4f}"
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

    mem_before = psutil.virtual_memory().used

    # Model training.
    print("Training...")
    train(device, g, dataset, num_classes, use_uva, fused_sampling, mem_before)