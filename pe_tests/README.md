# ogbn-products

## FBL
Setting
```
[20, 20, 20]
```
Peformance
```
lstime.mean (s): 0.2765758597583145
lsmem.mean (MB): 6.338676597582038
sampler memory (MB): 0.0
sampler comp (MB): 3670.046875
```
Setting
```
[10, 10, 10]
```
Performance
```
lstime.mean (s): 0.07474744958169308
lsmem.mean (MB): 4.704123488773748
Epoch 00002 | Loss 0.0000 | Time 14.0586
sampler memory (MB): -0.015625
sampler comp (MB): 2725.03125
```
Setting
```
[5, 5, 5]
```
Performance
```
lstime.mean (s): 0.01887392256544044
lsmem.mean (MB): 4.596313687392056
Epoch 00002 | Loss 0.0000 | Time 3.2730
sampler memory (MB): 0.0
sampler comp (MB): 2661.296875
```


## FCR

Setting
```
[20, 20, 20], alpha=2, T=50
```
Performance
```
time needed (s): 0.012237787246704102
memorage usage (MB): 0
lstime.mean (s): 0.25709364945406743
lsmem.mean (MB): 2.696216537132988
Epoch 00002 | Loss 0.0000 | Time 49.3013
sampler memory (MB): 5747.703125
sampler comp (MB): 7309.40625
```
Setting
```
[10, 10, 10], alpha=2, T=50
```
Performance
```
lstime.mean (s): 0.06386809027874409
lsmem.mean (MB): 2.11488018134715
Epoch 00002 | Loss 0.0000 | Time 12.0869
sampler memory (MB): 4754.109375
sampler comp (MB): 5980.1875
```
Setting
```
[5, 5, 5], alpha=2, T=50
```
Peformance
```
lstime.mean (s): 0.016260120741032155
lsmem.mean (MB): 1.2920984455958548
Epoch 00002 | Loss 0.0000 | Time 2.9867
sampler memory (MB): 3828.8125
sampler comp (MB): 4578.0
```
Setting
```
[20, 20, 20], alpha=1.5, T=50
```
Performance
```
time needed (s): 0.009011030197143555
memorage usage (MB): 0
lstime.mean (s): 0.2479564447682962
lsmem.mean (MB): 6.598904360967185
Epoch 00002 | Loss 0.0000 | Time 47.7072
sampler memory (MB): 5278.15625
sampler comp (MB): 9100.46875
```
Setting
```
[10, 10, 10], alpha=1.5, T=50
```
Performance
```
time needed (s): 0.0061130523681640625
memorage usage (MB): 0
lstime.mean (s): 0.0602550984252508
lsmem.mean (MB): 2.025340025906736
Epoch 00002 | Loss 0.0000 | Time 11.5165
sampler memory (MB): 4837.3125
sampler comp (MB): 6010.515625
```
Setting
```
[5, 5, 5], alpha=1.5, T=50
```
Performance
```
lstime.mean (s): 0.01521589801307375
lsmem.mean (MB): 1.8717076856649395
Epoch 00002 | Loss 0.0000 | Time 2.8404
sampler memory (MB): 3618.921875
sampler comp (MB): 4704.21875
```

# ogbn-arixv
## FBL
setting
```
sampler = NeighborSampler(
    [20, 20, 20],  # fanout for [layer-0, layer-1, layer-2]
    prefetch_node_feats=["feat"],
    prefetch_labels=["label"],
    fused=fused_sampling,
)
```
Performance
```
lstime.mean: 0.43414221542158377 s
lsmem.mean 116.24373829588015 MB
```

## FCR

arxiv
setting
```
sampler = NeighborSampler_FCR_struct(
    g=g,
    fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    alpha=2, T=50,
    prefetch_node_feats=["feat"],
    prefetch_labels=["label"],
    fused=fused_sampling,
)

```

```
lstime.mean: 0.5516397256529733 s
lsmem.mean: 21.298338014981272 MB
```

products
```
lstime.mean: 0.3341255570940403 s
lsmem.mean 6685.240150043178 MB
```


## FCR-SC

```
sampler = NeighborSampler_FCR_struct_shared_cache(
    g=g,
    fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    alpha=2, T=50,
    prefetch_node_feats=["feat"],
    prefetch_labels=["label"],
    fused=fused_sampling,
)
```

```
lstime.mean: 0.46770025639051804 s
lsmem.mean: 133.97237827715355 MB
```


## OTF
```
sampler = NeighborSampler_OTF_struct(
    g=g,
    fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [4,4,4]
    alpha=2, beta=1, gamma=0.15, T=358, #3, 0.4
    prefetch_node_feats=["feat"],
    prefetch_labels=["label"],
    fused=fused_sampling,
)
```

```
lstime.mean: 0.05882716878914051 s
lsmem.mean 6310.801840457686 MB
```

## OTF-SC
```
sampler = NeighborSampler_OTF_struct_shared_cache(
    g=g,
    fanouts=[20,20,20],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    alpha=2, beta=1, gamma=0.15, T=119,
    prefetch_node_feats=["feat"],
    prefetch_labels=["label"],
    fused=fused_sampling,
)
```

```
lstime.mean: 0.05370364535040188
lsmem.mean 3489.088622625216
```

# ogbn-mag


MultiLayerFullNeighborSampler,
NeighborSampler_FCR_struct_hete,
NeighborSampler_FCR_struct_shared_cache_hete,
NeighborSampler_OTF_refresh_struct_hete,
NeighborSampler_OTF_refresh_struct_shared_cache_hete,
NeighborSampler_OTF_fetch_struct_hete,
NeighborSampler_OTF_fetch_struct_shared_cache_hete,
NeighborSampler_OTF_struct_PCFPSCR_hete,
NeighborSampler_OTF_struct_PCFPSCR_shared_cache_hete,
NeighborSampler_OTF_struct_PSCRFCF_hete,
NeighborSampler_OTF_struct_PSCRFCF_shared_cache_hete,



FBL
```
Run: 01, Epoch: 01, Loss: 2.3219, Train: 62.44%, Valid: 49.37%, Test: 48.71%
Run: 01, Epoch: 02, Loss: 1.5387, Train: 75.45%, Valid: 49.61%, Test: 48.75%

Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:17<00:00, 4571.86it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:46<00:00, 6901.48it/s]
Run: 01, Epoch: 01, Loss: 2.4184, Train: 63.16%, Valid: 48.26%, Test: 47.06%
```

FCR
```
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:17<00:00, 4574.66it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:48<00:00, 6779.99it/s]
Run: 01, Epoch: 01, Loss: 2.3588, Train: 63.20%, Valid: 47.17%, Test: 46.55%
```
```
Run: 10, Epoch: 03, Loss: 1.0480, Train: 87.53%, Valid: 47.12%, Test: 45.85%
Run 10:
Highest Train: 87.53
Highest Valid: 49.35
  Final Train: 78.47
   Final Test: 48.23
Final performance: 
All runs:
Highest Train: 87.57 ± 0.22
Highest Valid: 48.31 ± 0.61
  Final Train: 75.66 ± 6.39
   Final Test: 47.30 ± 0.61
```

FCR - shared cache
```
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:13<00:00, 4729.16it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:43<00:00, 7096.64it/s]
Run: 01, Epoch: 01, Loss: 2.3702, Train: 63.04%, Valid: 47.98%, Test: 47.45%
```

OTF-refresh
```
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:17<00:00, 4571.86it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:46<00:00, 6901.48it/s]
Run: 01, Epoch: 01, Loss: 2.4184, Train: 63.16%, Valid: 48.26%, Test: 47.06%
```

OTF-refresh shared cache
```
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [02:24<00:00, 4361.53it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:48<00:00, 6757.76it/s]
Run: 01, Epoch: 01, Loss: 2.3566, Train: 63.41%, Valid: 49.16%, Test: 48.09%
```

OTF-fetch
```
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [03:17<00:00, 3189.88it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:46<00:00, 6893.49it/s]
Run: 01, Epoch: 01, Loss: 2.3918, Train: 61.69%, Valid: 48.43%, Test: 47.67%
```

OTF-fetch shared cache
```
Epoch 00: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [03:08<00:00, 3339.61it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:49<00:00, 6733.40it/s]
Run: 01, Epoch: 01, Loss: 2.4089, Train: 61.19%, Valid: 46.42%, Test: 45.77%
```

OTF-PCFPSCR
```
Epoch 00: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [29:59<00:00, 349.86it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:43<00:00, 7103.52it/s]
Run: 01, Epoch: 01, Loss: 2.3549, Train: 62.37%, Valid: 47.87%, Test: 47.30%
```

OTF-PCFPSCR shared cache
```
Epoch 00: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [36:14<00:00, 289.53it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:48<00:00, 6794.18it/s]
Run: 01, Epoch: 01, Loss: 2.3959, Train: 61.36%, Valid: 48.38%, Test: 48.15%
```

OTF-PSCRFCF
```
Epoch 00: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [28:00<00:00, 374.67it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:45<00:00, 6953.87it/s]
Run: 01, Epoch: 01, Loss: 2.3508, Train: 62.14%, Valid: 48.79%, Test: 48.00%
```

OTF-PSCRFCF-shared_cache
```
Epoch 00: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 629571/629571 [34:58<00:00, 300.01it/s]
Inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 736389/736389 [01:47<00:00, 6834.24it/s]
Run: 01, Epoch: 01, Loss: 2.3863, Train: 61.20%, Valid: 47.85%, Test: 47.12%
```
