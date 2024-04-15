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

## FCR-shared cache
Setting
```
[20, 20, 20], alpha=2, T=50
```
Performance
```
lstime.mean (s): 0.25541082070899135
lsmem.mean (MB): 4.428675518134715
sampler memory (MB): 1386.828125
sampler comp (MB): 3951.53125
```
Setting
```
[10, 10, 10], alpha=2, T=50
```
Performance
```
lstime.mean (s): 0.06397193989399595
lsmem.mean (MB): 4.627239853195164
Epoch 00002 | Loss 0.0000 | Time 12.3945
sampler memory (MB): 2744.125
sampler comp (MB): 5424.8125
```
Setting
```
[5, 5, 5], alpha=2, T=50
```
Performance
```
lstime.mean (s): 0.01607649931635881
lsmem.mean (MB): 1.6645077720207253
Epoch 00002 | Loss 0.0000 | Time 3.0865
sampler memory (MB): 2455.90625
sampler comp (MB): 3419.75
```
Setting
```
[20, 20, 20], alpha=1.5, T=50
```
Performance
```
lstime.mean (s): 0.24052175586087715
lsmem.mean (MB): 6.71362262521589
Epoch 00002 | Loss 0.0000 | Time 46.2107
sampler memory (MB): 527.3125
sampler comp (MB): 4414.84375
```
Setting
```
[10, 10, 10], alpha=1.5, T=50
```
Performance
```
lstime.mean (s): 0.059165482282226775
lsmem.mean (MB): 4.275340025906735
Epoch 00002 | Loss 0.0000 | Time 11.3112
sampler memory (MB): 362.125
sampler comp (MB): 2837.28125
```
Setting
```
[5, 5, 5], alpha=1.5, T=50
```
Performance
```
lstime.mean (s): 0.01488172328533904
lsmem.mean (MB): 0.10646049222797928
Epoch 00002 | Loss 0.0000 | Time 2.8331
sampler memory (MB): 1721.359375
sampler comp (MB): 1784.28125
```

## OTF
Setting
```
[20, 20, 20], amp_rate=2, refresh_rate=0.15, T=358
```
Performance
```
lstime.mean (s): 0.2459948853507561
lsmem.mean (MB): 4.13271804835924
Epoch 00002 | Loss 0.0000 | Time 47.2221
sampler memory (MB): 2657.890625
sampler comp (MB): 5052.359375
```
Setting
```
[20, 20, 20], amp_rate=2, refresh_rate=0.15, T=50
```
Performance
```
time needed (s): 0.005285024642944336
memorage usage (MB): 114688
lstime.mean (s): 0.2528338296425775
lsmem.mean (MB): 4.192357512953368
Epoch 00002 | Loss 0.0000 | Time 48.0913
sampler memory (MB): 2924.234375
sampler comp (MB): 5350.921875
```
Setting
```
[10, 10, 10], amp_rate=2, refresh_rate=0.15, T=50
```
Performance
```
time needed (s): 0.0026259422302246094
memorage usage (MB): 0
lstime.mean (s): 0.056848634298189524
lsmem.mean (MB): 1.8781573834196892
Epoch 00002 | Loss 0.0000 | Time 10.7541
sampler memory (MB): 1020.25
sampler comp (MB): 2108.046875
```
Setting
```
[5, 5, 5], amp_rate=2, refresh_rate=0.15, T=50
```
Performance
```
time needed (s): 0.0018260478973388672
memorage usage (MB): 0
lstime.mean (s): 0.014533051966797297
lsmem.mean (MB): 0.32070379965457685
Epoch 00002 | Loss 0.0000 | Time 2.6287
sampler memory (MB): 1240.4375
sampler comp (MB): 1427.65625
```
Setting
```
[20, 20, 20], amp_rate=1.5, refresh_rate=0.15, T=50
```
Performance
```
time needed (s): 0.005866050720214844
memorage usage (MB): 0
lstime.mean (s): 0.2424830072696139
lsmem.mean (MB): 2.588784542314335
Epoch 00002 | Loss 0.0000 | Time 44.9782
sampler memory (MB): 2510.609375
sampler comp (MB): 4016.390625
```
Setting
```
[10, 10, 10], amp_rate=1.5, refresh_rate=0.15, T=50
```
Performance
```
time needed (s): 0.0029630661010742188
memorage usage (MB): 0
lstime.mean (s): 0.055565218027803356
lsmem.mean (MB): 2.7754209844559585
Epoch 00002 | Loss 0.0000 | Time 10.6695
sampler memory (MB): 1660.0625
sampler comp (MB): 3265.4375
```
Setting
```
[5, 5, 5], amp_rate=1.5, refresh_rate=0.15, T=50
```
Performance
```
time needed (s): 0.0015718936920166016
memorage usage (MB): 0
lstime.mean (s): 0.014085919951733736
lsmem.mean (MB): -0.6624028497409327
Epoch 00002 | Loss 0.0000 | Time 2.6370
sampler memory (MB): 794.359375
sampler comp (MB): 412.40625
```
Setting
```
[20, 20, 20], amp_rate=1.5, refresh_rate=0.3, T=50
```
Performance
```
time needed (s): 0.0057260990142822266
memorage usage (MB): 196608
lstime.mean (s): 0.24946731771623945
lsmem.mean (MB): 3.8385956390328153
Epoch 00002 | Loss 0.0000 | Time 49.0388
sampler memory (MB): 2818.84375
sampler comp (MB): 5042.765625
```
Setting
```
[10, 10, 10], amp_rate=1.5, refresh_rate=0.3, T=50
```
Performance
```
lstime.mean (s): 0.05582910498189185
lsmem.mean (MB): 1.56452396373057
Epoch 00002 | Loss 0.0000 | Time 10.4803
sampler memory (MB): 1655.46875
sampler comp (MB): 2562.9375
```
Setting
```
[5, 5, 5], amp_rate=1.5, refresh_rate=0.3, T=50
lstime.mean (s): 0.01460982686703275
lsmem.mean (MB): -0.3279360967184801
Epoch 00002 | Loss 0.0000 | Time 2.6154
sampler memory (MB): 1102.09375
sampler comp (MB): 922.140625
```

## OTF - shared cache
Setting
```
[20, 20, 20], amp_rate=2, refresh_rate=0.15, T=50
```
Performance
```
lstime.mean (s): 0.23931460347612082
lsmem.mean (MB): 0.6762197754749568
Epoch 00002 | Loss 0.0000 | Time 46.0906
sampler memory (MB): 2423.484375
sampler comp (MB): 2816.703125
```
Setting
```
[10, 10, 10], amp_rate=2, refresh_rate=0.15, T=50
```
Performance
```
lstime.mean (s): 0.05590548869447593
lsmem.mean (MB): 0.8663104490500864
Epoch 00002 | Loss 0.0000 | Time 10.7997
sampler memory (MB): 2064.453125
sampler comp (MB): 2567.640625
```
Setting
```
[5, 5, 5], amp_rate=2, refresh_rate=0.15, T=50
```
Performance
```
lstime.mean (s): 0.013263786394979053
lsmem.mean (MB): 1.414885578583765
Epoch 00002 | Loss 0.0000 | Time 2.5175
sampler memory (MB): 76.578125
sampler comp (MB): 896.125
```
Setting
```
[20, 20, 20], amp_rate=1.5, refresh_rate=0.15, T=50
```
Performance
```
lstime.mean (s): 0.2326089779947706
lsmem.mean (MB): 0.5644969775474957
Epoch 00002 | Loss 0.0000 | Time 44.9933
sampler memory (MB): 2316.125
sampler comp (MB): 2644.6875
```
Setting
```
[10, 10, 10], amp_rate=1.5, refresh_rate=0.15, T=50
```
Performance
lstime.mean (s): 0.05265655418751771
lsmem.mean (MB): 2.6571135578583767
Epoch 00002 | Loss 0.0000 | Time 10.1088
sampler memory (MB): 252.421875
sampler comp (MB): 1799.0
```
Setting
```
[5, 5, 5], amp_rate=1.5, refresh_rate=0.15, T=50
```
Performance
```
lstime.mean (s): 0.012637567849562782
lsmem.mean (MB): -1.0073402417962003
Epoch 00002 | Loss 0.0000 | Time 2.4326
sampler memory (MB): 1731.453125
sampler comp (MB): 1149.75
```
Setting
```
[20, 20, 20], amp_rate=2, refresh_rate=0.3, T=50
```
Performance
```
lstime.mean (s): 0.2389718477384208
lsmem.mean (MB): 3.636846934369603
Epoch 00002 | Loss 0.0000 | Time 46.0592
sampler memory (MB): 967.9375
sampler comp (MB): 3074.875
```
Setting
```
[10, 10, 10], amp_rate=2, refresh_rate=0.3, T=50
```
Performance
```
lstime.mean (s): 0.05598670295688978
lsmem.mean (MB): 2.7602277633851466
Epoch 00002 | Loss 0.0000 | Time 10.7512
sampler memory (MB): 447.375
sampler comp (MB): 2044.171875
```
Setting
```
[5, 5, 5], amp_rate=2, refresh_rate=0.3, T=50
```
Performance
```
lstime.mean (s): 0.013321832877572543
lsmem.mean (MB): 0.03181670984455959
Epoch 00002 | Loss 0.0000 | Time 2.5416
sampler memory (MB): 1819.390625
sampler comp (MB): 1837.3125
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
