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
    NeighborSampler_FCR_struct_shared_cache,
    NeighborSampler_OTF_struct_FSCRFCF,
    NeighborSampler_OTF_struct_FSCRFCF_shared_cache,
    NeighborSampler_OTF_struct_PCFFSCR_shared_cache,
    NeighborSampler_OTF_struct_PCFFSCR,
    NeighborSampler_OTF_struct_PCFPSCR_SC,
    NeighborSampler_OTF_struct_PCFPSCR,
    # NeighborSampler_OTF_struct,
    # NeighborSampler_OTF_struct_shared_cache

)
from ogb.nodeproppred import DglNodePropPredDataset

def train(device, g, dataset, num_classes, use_uva, fused_sampling, mem_before):
    # Create sampler & dataloader.
    train_idx = dataset.train_idx.to(g.device if not use_uva else device)
    val_idx = dataset.val_idx.to(g.device if not use_uva else device)

    mem_pre_sample = psutil.virtual_memory().used

    # # FBL
    # sampler = NeighborSampler(
    #     [20, 20, 20],  # fanout for [layer-0, layer-1, layer-2]
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # # """
    # # lstime.mean (s): 0.2765758597583145
    # # lsmem.mean (MB): 6.338676597582038
    # # sampler memory (MB): 0.0
    # # sampler comp (MB): 3670.046875
    # # """

    # # FCR
    # sampler = NeighborSampler_FCR_struct(
    #     g=g,
    #     fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     alpha=2, T=50,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # # """
    # # lstime.mean (s): 0.25917509965122465
    # # lsmem.mean (MB): 2.0676273747841107
    # # sampler memory (MB): 3688.75
    # # sampler comp (MB): 4886.546875
    # # """

    # # FCR shared cache
    # sampler = NeighborSampler_FCR_struct_shared_cache(
    #     g=g,
    #     fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     alpha=2, T=50,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )    

    # # """
    # # lstime.mean (s): 0.25541082070899135
    # # lsmem.mean (MB): 4.428675518134715
    # # sampler memory (MB): 1386.828125
    # # sampler comp (MB): 3951.53125
    # # """

    # # OTF
    # sampler = NeighborSampler_OTF_struct_FSCRFCF(
    #     g=g,
    #     fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [4,4,4]
    #     amp_rate=2, refresh_rate=0.15, T=358, #3, 0.4
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # # """
    # # lstime.mean (s): 0.2459948853507561
    # # lsmem.mean (MB): 4.13271804835924
    # # Epoch 00002 | Loss 0.0000 | Time 47.2221
    # # sampler memory (MB): 2657.890625
    # # sampler comp (MB): 5052.359375
    # # """

    # # OTF shared cache
    # sampler = NeighborSampler_OTF_struct_FSCRFCF_shared_cache(
    #     g=g,
    #     fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     alpha=2, beta=1, gamma=0.15, T=119,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # # """
    # # lstime.mean (s): 0.25199997363312876
    # # lsmem.mean (MB): 1.6715511658031088
    # # Epoch 00002 | Loss 0.0000 | Time 47.5013
    # # sampler memory (MB): 841.296875
    # # sampler comp (MB): 1812.578125
    # # """

    # # OTF FSCR FCF shared cache
    # sampler = NeighborSampler_OTF_struct_PCFFSCR_shared_cache(
    #     g=g,
    #     fanouts=[20,20,20],
    #     amp_rate=1.5,fetch_rate=0.4,T_fetch=10
    # )

    # # """
    # # lstime.mean (s): 0.3299847666257815
    # # lsmem.mean (MB): 3.713811528497409
    # # Epoch 00002 | Loss 0.0000 | Time 62.2927
    # # sampler memory (MB): 444.21875
    # # sampler comp (MB): 1006.125
    # # """

    # # OTF FSCR FCF
    # sampler = NeighborSampler_OTF_struct_PCFFSCR(
    #     g=g,
    #     fanouts=[20,20,20],
    #     amp_rate=1.5,fetch_rate=0.4,T_fetch=10
    # )

    # """
    # time needed (s): 0.006249189376831055
    # memorage usage (MB): 0
    # lstime.mean (s): 0.38904782651001923
    # lsmem.mean (MB): -8.920633635578584
    # Epoch 00002 | Loss 0.0000 | Time 76.9332
    # sampler memory (MB): 2604.015625
    # sampler comp (MB): -2566.578125
    # """

    # PCF PSCR SC
    # sampler = NeighborSampler_OTF_struct_PCFPSCR_SC(
    #     g=g,
    #     fanouts=[20,20,20],
    #     amp_rate=1.5,refresh_rate=0.4,T=10
    # )

    # PCF PSCR
    sampler = NeighborSampler_OTF_struct_PCFPSCR(
        g=g,
        fanouts=[20,20,20],
        amp_rate=1.5,refresh_rate=0.4,T=10
    )



    mem_after_sample = psutil.virtual_memory().used

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
        mem_before = psutil.virtual_memory().used
        # Before the operation
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # After the operation
            mem_after = psutil.virtual_memory().used
            # memusage = process.memory_info().rss / (1024 ** 3) 
            t2=time.time()
            # lsmem.append(memusage)
            lstime.append(t2-t20)
            # print("memorage usage (GB):",memusage)

            memusage = mem_after-mem_before
            lsmem.append(memusage/(1024**2))
            print("time needed (s):",t2-t20)
            print("memorage usage (MB):",memusage)
            # Before the operation
            mem_before = psutil.virtual_memory().used
            t20=time.time()
        t1 = time.time()
        print("lstime.mean (s):",sum(lstime)/len(lstime))
        print("lsmem.mean (MB):",sum(lsmem)/len(lsmem))
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | Time {t1 - t0:.4f}"
        )
    sampler_mem = (mem_after_sample- mem_pre_sample)/(1024 ** 2)
    mem_after_compute = psutil.virtual_memory().used
    sampler_comp = (mem_after_compute-mem_pre_sample)/(1024**2)
    print("sampler memory (MB):", sampler_mem)
    print("sampler comp (MB):", sampler_comp)

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