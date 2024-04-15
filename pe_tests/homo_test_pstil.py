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
    NeighborSampler_OTF_struct_PSCRFCF_SC,
    NeighborSampler_OTF_struct_PSCRFCF,
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
    #     [5, 5, 5],  # fanout for [layer-0, layer-1, layer-2]
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # """
    # [20, 20, 20]
    # lstime.mean (s): 0.2765758597583145
    # lsmem.mean (MB): 6.338676597582038
    # sampler memory (MB): 0.0
    # sampler comp (MB): 3670.046875
    # """
    
    # """
    # [10, 10, 10]
    # lstime.mean (s): 0.07474744958169308
    # lsmem.mean (MB): 4.704123488773748
    # Epoch 00002 | Loss 0.0000 | Time 14.0586
    # sampler memory (MB): -0.015625
    # sampler comp (MB): 2725.03125
    # """

    # """
    # [5, 5, 5]
    # lstime.mean (s): 0.01887392256544044
    # lsmem.mean (MB): 4.596313687392056
    # Epoch 00002 | Loss 0.0000 | Time 3.2730
    # sampler memory (MB): 0.0
    # sampler comp (MB): 2661.296875
    # """

    # # FCR
    # sampler = NeighborSampler_FCR_struct(
    #     g=g,
    #     fanouts=[5,5,5],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     alpha=1.5, T=50,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # """
    # [20, 20, 20], alpha=2, T=50
    # time needed (s): 0.012237787246704102
    # memorage usage (MB): 0
    # lstime.mean (s): 0.25709364945406743
    # lsmem.mean (MB): 2.696216537132988
    # Epoch 00002 | Loss 0.0000 | Time 49.3013
    # sampler memory (MB): 5747.703125
    # sampler comp (MB): 7309.40625
    # """

    # """
    # [10, 10, 10], alpha=2, T=50
    # lstime.mean (s): 0.06386809027874409
    # lsmem.mean (MB): 2.11488018134715
    # Epoch 00002 | Loss 0.0000 | Time 12.0869
    # sampler memory (MB): 4754.109375
    # sampler comp (MB): 5980.1875
    # """

    # """
    # lstime.mean (s): 0.016260120741032155
    # lsmem.mean (MB): 1.2920984455958548
    # Epoch 00002 | Loss 0.0000 | Time 2.9867
    # sampler memory (MB): 3828.8125
    # sampler comp (MB): 4578.0
    # """

    # """
    # [20, 20, 20], alpha=1.5, T=50
    # time needed (s): 0.009011030197143555
    # memorage usage (MB): 0
    # lstime.mean (s): 0.2479564447682962
    # lsmem.mean (MB): 6.598904360967185
    # Epoch 00002 | Loss 0.0000 | Time 47.7072
    # sampler memory (MB): 5278.15625
    # sampler comp (MB): 9100.46875
    # """

    # """
    # [10, 10, 10], alpha=1.5, T=50
    # time needed (s): 0.0061130523681640625
    # memorage usage (MB): 0
    # lstime.mean (s): 0.0602550984252508
    # lsmem.mean (MB): 2.025340025906736
    # Epoch 00002 | Loss 0.0000 | Time 11.5165
    # sampler memory (MB): 4837.3125
    # sampler comp (MB): 6010.515625
    # """

    # """
    # [5, 5, 5], alpha=1.5, T=50
    # lstime.mean (s): 0.01521589801307375
    # lsmem.mean (MB): 1.8717076856649395
    # Epoch 00002 | Loss 0.0000 | Time 2.8404
    # sampler memory (MB): 3618.921875
    # sampler comp (MB): 4704.21875
    # """

    # # FCR shared cache
    # sampler = NeighborSampler_FCR_struct_shared_cache(
    #     g=g,
    #     fanouts=[5,5,5],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     alpha=1.5, T=50,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )    

    # """
    # [20, 20, 20], alpha=2, T=50
    # lstime.mean (s): 0.25541082070899135
    # lsmem.mean (MB): 4.428675518134715
    # sampler memory (MB): 1386.828125
    # sampler comp (MB): 3951.53125
    # """

    # """
    # [10, 10, 10], alpha=2, T=50
    # lstime.mean (s): 0.06397193989399595
    # lsmem.mean (MB): 4.627239853195164
    # Epoch 00002 | Loss 0.0000 | Time 12.3945
    # sampler memory (MB): 2744.125
    # sampler comp (MB): 5424.8125
    # """

    # """
    # [5, 5, 5], alpha=2, T=50
    # lstime.mean (s): 0.01607649931635881
    # lsmem.mean (MB): 1.6645077720207253
    # Epoch 00002 | Loss 0.0000 | Time 3.0865
    # sampler memory (MB): 2455.90625
    # sampler comp (MB): 3419.75
    # """

    # """
    # lstime.mean (s): 0.24052175586087715
    # lsmem.mean (MB): 6.71362262521589
    # Epoch 00002 | Loss 0.0000 | Time 46.2107
    # sampler memory (MB): 527.3125
    # sampler comp (MB): 4414.84375
    # """

    # """
    # [10, 10, 10], alpha=1.5, T=50
    # lstime.mean (s): 0.059165482282226775
    # lsmem.mean (MB): 4.275340025906735
    # Epoch 00002 | Loss 0.0000 | Time 11.3112
    # sampler memory (MB): 362.125
    # sampler comp (MB): 2837.28125
    # """

    # """
    # [5, 5, 5], alpha=1.5, T=50
    # lstime.mean (s): 0.01488172328533904
    # lsmem.mean (MB): 0.10646049222797928
    # Epoch 00002 | Loss 0.0000 | Time 2.8331
    # sampler memory (MB): 1721.359375
    # sampler comp (MB): 1784.28125
    # """

    # # OTF
    # sampler = NeighborSampler_OTF_struct_FSCRFCF(
    #     g=g,
    #     fanouts=[5,5,5],  # fanout for [layer-0, layer-1, layer-2] [4,4,4]
    #     amp_rate=2, refresh_rate=0.3, T=50, #3, 0.4
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )

    # """
    # [20, 20, 20], amp_rate=2, refresh_rate=0.15, T=358
    # lstime.mean (s): 0.2459948853507561
    # lsmem.mean (MB): 4.13271804835924
    # Epoch 00002 | Loss 0.0000 | Time 47.2221
    # sampler memory (MB): 2657.890625
    # sampler comp (MB): 5052.359375
    # """

    # """
    # [20, 20, 20], amp_rate=2, refresh_rate=0.15, T=50
    # time needed (s): 0.005285024642944336
    # memorage usage (MB): 114688
    # lstime.mean (s): 0.2528338296425775
    # lsmem.mean (MB): 4.192357512953368
    # Epoch 00002 | Loss 0.0000 | Time 48.0913
    # sampler memory (MB): 2924.234375
    # sampler comp (MB): 5350.921875
    # """

    # """
    # [10, 10, 10], amp_rate=2, refresh_rate=0.15, T=50
    # time needed (s): 0.0026259422302246094
    # memorage usage (MB): 0
    # lstime.mean (s): 0.056848634298189524
    # lsmem.mean (MB): 1.8781573834196892
    # Epoch 00002 | Loss 0.0000 | Time 10.7541
    # sampler memory (MB): 1020.25
    # sampler comp (MB): 2108.046875
    # """

    # """
    # [5, 5, 5], amp_rate=2, refresh_rate=0.15, T=50
    # time needed (s): 0.0018260478973388672
    # memorage usage (MB): 0
    # lstime.mean (s): 0.014533051966797297
    # lsmem.mean (MB): 0.32070379965457685
    # Epoch 00002 | Loss 0.0000 | Time 2.6287
    # sampler memory (MB): 1240.4375
    # sampler comp (MB): 1427.65625
    # """

    # """
    # [20, 20, 20], amp_rate=1.5, refresh_rate=0.15, T=50
    # time needed (s): 0.005866050720214844
    # memorage usage (MB): 0
    # lstime.mean (s): 0.2424830072696139
    # lsmem.mean (MB): 2.588784542314335
    # Epoch 00002 | Loss 0.0000 | Time 44.9782
    # sampler memory (MB): 2510.609375
    # sampler comp (MB): 4016.390625
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, refresh_rate=0.15, T=50
    # time needed (s): 0.0029630661010742188
    # memorage usage (MB): 0
    # lstime.mean (s): 0.055565218027803356
    # lsmem.mean (MB): 2.7754209844559585
    # Epoch 00002 | Loss 0.0000 | Time 10.6695
    # sampler memory (MB): 1660.0625
    # sampler comp (MB): 3265.4375
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, refresh_rate=0.15, T=50
    # time needed (s): 0.0015718936920166016
    # memorage usage (MB): 0
    # lstime.mean (s): 0.014085919951733736
    # lsmem.mean (MB): -0.6624028497409327
    # Epoch 00002 | Loss 0.0000 | Time 2.6370
    # sampler memory (MB): 794.359375
    # sampler comp (MB): 412.40625
    # """

    # """
    # [20, 20, 20], amp_rate=1.5, refresh_rate=0.3, T=50
    # time needed (s): 0.0057260990142822266
    # memorage usage (MB): 196608
    # lstime.mean (s): 0.24946731771623945
    # lsmem.mean (MB): 3.8385956390328153
    # Epoch 00002 | Loss 0.0000 | Time 49.0388
    # sampler memory (MB): 2818.84375
    # sampler comp (MB): 5042.765625
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, refresh_rate=0.3, T=50
    # lstime.mean (s): 0.05582910498189185
    # lsmem.mean (MB): 1.56452396373057
    # Epoch 00002 | Loss 0.0000 | Time 10.4803
    # sampler memory (MB): 1655.46875
    # sampler comp (MB): 2562.9375
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, refresh_rate=0.3, T=50
    # lstime.mean (s): 0.01460982686703275
    # lsmem.mean (MB): -0.3279360967184801
    # Epoch 00002 | Loss 0.0000 | Time 2.6154
    # sampler memory (MB): 1102.09375
    # sampler comp (MB): 922.140625
    # """

    # # OTF shared cache
    # sampler = NeighborSampler_OTF_struct_FSCRFCF_shared_cache(
    #     g=g,
    #     fanouts=[5,5,5],  # fanout for [layer-0, layer-1, layer-2] [2,2,2]
    #     # alpha=2, beta=1, gamma=0.15, T=119,
    #     amp_rate=2, refresh_rate=0.3, T=50,
    #     prefetch_node_feats=["feat"],
    #     prefetch_labels=["label"],
    #     fused=fused_sampling,
    # )
    
    # """
    # [20, 20, 20], amp_rate=2, refresh_rate=0.15, T=50
    # lstime.mean (s): 0.23931460347612082
    # lsmem.mean (MB): 0.6762197754749568
    # Epoch 00002 | Loss 0.0000 | Time 46.0906
    # sampler memory (MB): 2423.484375
    # sampler comp (MB): 2816.703125
    # """

    # """
    # [10, 10, 10], amp_rate=2, refresh_rate=0.15, T=50
    # lstime.mean (s): 0.05590548869447593
    # lsmem.mean (MB): 0.8663104490500864
    # Epoch 00002 | Loss 0.0000 | Time 10.7997
    # sampler memory (MB): 2064.453125
    # sampler comp (MB): 2567.640625
    # """

    # """
    # [5, 5, 5], amp_rate=2, refresh_rate=0.15, T=50
    # lstime.mean (s): 0.013263786394979053
    # lsmem.mean (MB): 1.414885578583765
    # Epoch 00002 | Loss 0.0000 | Time 2.5175
    # sampler memory (MB): 76.578125
    # sampler comp (MB): 896.125
    # """

    # """
    # [20, 20, 20], amp_rate=1.5, refresh_rate=0.15, T=50
    # lstime.mean (s): 0.2326089779947706
    # lsmem.mean (MB): 0.5644969775474957
    # Epoch 00002 | Loss 0.0000 | Time 44.9933
    # sampler memory (MB): 2316.125
    # sampler comp (MB): 2644.6875
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, refresh_rate=0.15, T=50
    # lstime.mean (s): 0.05265655418751771
    # lsmem.mean (MB): 2.6571135578583767
    # Epoch 00002 | Loss 0.0000 | Time 10.1088
    # sampler memory (MB): 252.421875
    # sampler comp (MB): 1799.0
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, refresh_rate=0.15, T=50
    # lstime.mean (s): 0.012637567849562782
    # lsmem.mean (MB): -1.0073402417962003
    # Epoch 00002 | Loss 0.0000 | Time 2.4326
    # sampler memory (MB): 1731.453125
    # sampler comp (MB): 1149.75
    # """

    # """
    # [20, 20, 20], amp_rate=2, refresh_rate=0.3, T=50
    # lstime.mean (s): 0.2389718477384208
    # lsmem.mean (MB): 3.636846934369603
    # Epoch 00002 | Loss 0.0000 | Time 46.0592
    # sampler memory (MB): 967.9375
    # sampler comp (MB): 3074.875
    # """

    # """
    # [10, 10, 10], amp_rate=2, refresh_rate=0.3, T=50
    # lstime.mean (s): 0.05598670295688978
    # lsmem.mean (MB): 2.7602277633851466
    # Epoch 00002 | Loss 0.0000 | Time 10.7512
    # sampler memory (MB): 447.375
    # sampler comp (MB): 2044.171875
    # """

    # """
    # [5, 5, 5], amp_rate=2, refresh_rate=0.3, T=50
    # lstime.mean (s): 0.013321832877572543
    # lsmem.mean (MB): 0.03181670984455959
    # Epoch 00002 | Loss 0.0000 | Time 2.5416
    # sampler memory (MB): 1819.390625
    # sampler comp (MB): 1837.3125
    # """


    # # OTF FSCR FCF shared cache
    # sampler = NeighborSampler_OTF_struct_PCFFSCR_shared_cache(
    #     g=g,
    #     fanouts=[5,5,5],
    #     amp_rate=2,fetch_rate=0.3,T_fetch=10
    # )

    # """
    # [20, 20, 20], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.3026451702150862
    # lsmem.mean (MB): 3.7811150690846285
    # Epoch 00002 | Loss 0.0000 | Time 57.7882
    # sampler memory (MB): 869.71875
    # sampler comp (MB): 3057.296875
    # """

    # """
    # [10, 10, 10], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.10660997244154645
    # lsmem.mean (MB): 0.09412780656303972
    # Epoch 00002 | Loss 0.0000 | Time 20.5457
    # sampler memory (MB): 2072.890625
    # sampler comp (MB): 2129.0625
    # """

    # """
    # [5, 5, 5], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.06275247498085668
    # lsmem.mean (MB): 0.9979490500863558
    # Epoch 00002 | Loss 0.0000 | Time 12.0080
    # sampler memory (MB): 629.8125
    # sampler comp (MB): 1208.21875
    # """

    # """
    # [20, 20, 20], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # lstime.mean (s): 0.3299847666257815
    # lsmem.mean (MB): 3.713811528497409
    # Epoch 00002 | Loss 0.0000 | Time 62.2927
    # sampler memory (MB): 444.21875
    # sampler comp (MB): 1006.125
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # lstime.mean (s): 0.10309383041500428
    # lsmem.mean (MB): 2.3821243523316062
    # Epoch 00002 | Loss 0.0000 | Time 19.8745
    # sampler memory (MB): 472.0
    # sampler comp (MB): 1851.890625
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # lstime.mean (s): 0.06141914275437851
    # lsmem.mean (MB): 0.9039291882556131
    # Epoch 00002 | Loss 0.0000 | Time 11.7546
    # sampler memory (MB): 78.75
    # sampler comp (MB): 602.203125
    # """

    # # OTF FSCR FCF
    # sampler = NeighborSampler_OTF_struct_PCFFSCR(
    #     g=g,
    #     fanouts=[5,5,5],
    #     amp_rate=2,fetch_rate=0.3,T_fetch=10
    # )

    # """
    # [20, 20, 20], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # time needed (s): 0.006249189376831055
    # memorage usage (MB): 0
    # lstime.mean (s): 0.38904782651001923
    # lsmem.mean (MB): -8.920633635578584
    # Epoch 00002 | Loss 0.0000 | Time 76.9332
    # sampler memory (MB): 2604.015625
    # sampler comp (MB): -2566.578125
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # lstime.mean (s): 0.10557236119667278
    # lsmem.mean (MB): 2.5692465457685665
    # Epoch 00002 | Loss 0.0000 | Time 20.1938
    # sampler memory (MB): 1727.59375
    # sampler comp (MB): 3216.875
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # lstime.mean (s): 0.06438153889512768
    # lsmem.mean (MB): 0.8608322538860104
    # Epoch 00002 | Loss 0.0000 | Time 12.3647
    # sampler memory (MB): 1090.953125
    # sampler comp (MB): 1589.96875
    # """

    # """
    # [20, 20, 20], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.2892284520961254
    # lsmem.mean (MB): 1.3424816493955094
    # Epoch 00002 | Loss 0.0000 | Time 55.4870
    # sampler memory (MB): 4554.4375
    # sampler comp (MB): 5333.25
    # """

    # """
    # [10, 10, 10], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.11041197957976083
    # lsmem.mean (MB): 2.9305375647668392
    # Epoch 00002 | Loss 0.0000 | Time 21.0894
    # sampler memory (MB): 3284.25
    # sampler comp (MB): 4981.578125
    # """

    # """
    # [5, 5, 5], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.064887913603445
    # lsmem.mean (MB): 1.180510578583765
    # Epoch 00002 | Loss 0.0000 | Time 12.5179
    # sampler memory (MB): 1313.5625
    # sampler comp (MB): 1998.59375
    # """

    # # PCF PSCR SC
    # sampler = NeighborSampler_OTF_struct_PCFPSCR_SC(
    #     g=g,
    #     fanouts=[5,5,5],
    #     amp_rate=2,refresh_rate=0.3,T=10
    # )

    # """
    # [20, 20, 20], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # lstime.mean (s): 0.6796785866864193
    # lsmem.mean (MB): 14.831741148531952
    # Epoch 00002 | Loss 0.0000 | Time 130.4356
    # sampler memory (MB): 3283.890625
    # sampler comp (MB): 11876.671875
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # lstime.mean (s): 0.39627794568814556
    # lsmem.mean (MB): 12.67630073402418
    # Epoch 00002 | Loss 0.0000 | Time 75.8541
    # sampler memory (MB): 1304.078125
    # sampler comp (MB): 8646.03125
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, fetch_rate=0.4, T_fetch=10
    # lstime.mean (s): 0.2563990401892258
    # lsmem.mean (MB): 5.410891623488774
    # Epoch 00002 | Loss 0.0000 | Time 49.3908
    # sampler memory (MB): 1741.734375
    # sampler comp (MB): 4876.234375
    # """

    # """
    # [20, 20, 20], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.8273224336495671
    # lsmem.mean (MB): 15.331768134715025
    # Epoch 00002 | Loss 0.0000 | Time 159.9508
    # sampler memory (MB): 2477.234375
    # sampler comp (MB): 11361.40625
    # """

    # """
    # [10, 10, 10], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.4559945067799359
    # lsmem.mean (MB): 16.605677892918827
    # Epoch 00002 | Loss 0.0000 | Time 87.5929
    # sampler memory (MB): 497.90625
    # sampler comp (MB): 10131.078125
    # """

    # """
    # [5, 5, 5], amp_rate=2, fetch_rate=0.3, T_fetch=10
    # lstime.mean (s): 0.2931014754405705
    # lsmem.mean (MB): 12.777283031088082
    # Epoch 00002 | Loss 0.0000 | Time 56.4575
    # sampler memory (MB): 263.515625
    # sampler comp (MB): 7662.4375
    # """


    # # PCF PSCR
    # sampler = NeighborSampler_OTF_struct_PCFPSCR(
    #     g=g,
    #     fanouts=[5,5,5],
    #     amp_rate=2,refresh_rate=0.3,T=50
    # )
    # """
    # [20, 20, 20], amp_rate=1.5, fetch_rate=0.4, T_fetch=50
    # lstime.mean (s): 0.31865267061816593
    # lsmem.mean (MB): 16.5578853626943
    # Epoch 00002 | Loss 0.0000 | Time 62.8892
    # sampler memory (MB): 2826.71875
    # sampler comp (MB): 12415.28125
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, fetch_rate=0.4, T_fetch=50
    # lstime.mean (s): 0.12711157699940737
    # lsmem.mean (MB): 14.049573618307427
    # Epoch 00002 | Loss 0.0000 | Time 25.9544
    # sampler memory (MB): 1728.109375
    # sampler comp (MB): 9862.8125
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, fetch_rate=0.4, T_fetch=50
    # lstime.mean (s): 0.06914033090924146
    # lsmem.mean (MB): 8.573240500863557
    # Epoch 00002 | Loss 0.0000 | Time 14.3358
    # sampler memory (MB): 1295.640625
    # sampler comp (MB): 6261.109375
    # """

    # """
    # [20, 20, 20], amp_rate=2, fetch_rate=0.3, T_fetch=50
    # lstime.mean (s): 0.3454093690157348
    # lsmem.mean (MB): 18.001996977547496
    # Epoch 00002 | Loss 0.0000 | Time 69.0240
    # sampler memory (MB): 3104.96875
    # sampler comp (MB): 13528.109375
    # """

    # """
    # [10, 10, 10], amp_rate=2, fetch_rate=0.3, T_fetch=50
    # lstime.mean (s): 0.14107460185036141
    # lsmem.mean (MB): 13.733673359240068
    # Epoch 00002 | Loss 0.0000 | Time 28.8773
    # sampler memory (MB): 3321.296875
    # sampler comp (MB): 11277.15625
    # """

    # """
    # [5, 5, 5], amp_rate=2, fetch_rate=0.3, T_fetch=50
    # lstime.mean (s): 0.07714809572552564
    # lsmem.mean (MB): 10.41985103626943
    # Epoch 00002 | Loss 0.0000 | Time 16.0195
    # sampler memory (MB): 1615.703125
    # sampler comp (MB): 7650.328125
    # """

    # # PSCR FCF SC
    # sampler = NeighborSampler_OTF_struct_PSCRFCF_SC(
    #     g=g,
    #     fanouts=[5,5,5],
    #     amp_rate=2, refresh_rate=0.3, T=50
    # )

    # """
    # [20, 20, 20], amp_rate=1.5, fetch_rate=0.4, T_fetch=50
    # lstime.mean (s): 0.32016538214807067
    # lsmem.mean (MB): 14.088730569948186
    # Epoch 00002 | Loss 0.0000 | Time 64.6227
    # sampler memory (MB): 1571.0625
    # sampler comp (MB): 9730.171875
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, fetch_rate=0.4, T_fetch=50
    # lstime.mean (s): 0.12137314198548312
    # lsmem.mean (MB): 10.093210276338514
    # Epoch 00002 | Loss 0.0000 | Time 24.4835
    # sampler memory (MB): 1962.359375
    # sampler comp (MB): 7815.875
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, fetch_rate=0.4, T_fetch=50
    # lstime.mean (s): 0.05846894549170512
    # lsmem.mean (MB): 4.396831822107081
    # Epoch 00002 | Loss 0.0000 | Time 11.8837
    # sampler memory (MB): 1220.25
    # sampler comp (MB): 3766.578125
    # """

    # """
    # [20, 20, 20], amp_rate=2, fetch_rate=0.3, T_fetch=50
    # lstime.mean (s): 0.35212161833346406
    # lsmem.mean (MB): 15.586328799654577
    # Epoch 00002 | Loss 0.0000 | Time 68.9015
    # sampler memory (MB): 1526.328125
    # sampler comp (MB): 10551.671875
    # """

    # """
    # [10, 10, 10], amp_rate=2, fetch_rate=0.3, T_fetch=50
    # lstime.mean (s): 0.1308849842025414
    # lsmem.mean (MB): 13.403119602763384
    # Epoch 00002 | Loss 0.0000 | Time 26.1022
    # sampler memory (MB): 1134.34375
    # sampler comp (MB): 8896.5
    # """

    # """
    # [5, 5, 5], amp_rate=2, fetch_rate=0.3, T_fetch=50
    # lstime.mean (s): 0.06724883608249803
    # lsmem.mean (MB): 7.4897452504317785
    # Epoch 00002 | Loss 0.0000 | Time 13.7832
    # sampler memory (MB): 867.4375
    # sampler comp (MB): 5200.734375
    # """

    # PSCR FCF
    sampler = NeighborSampler_OTF_struct_PSCRFCF(
        g=g,
        fanouts=[5,5,5],
        amp_rate=2,
        refresh_rate=0.3,
        T=50,
    )

    # """
    # [20, 20, 20], amp_rate=2, refresh_rate=0.3, T=50
    # lstime.mean (s): 0.3558487307428284
    # lsmem.mean (MB): 21.278308506044905
    # Epoch 00002 | Loss 0.0000 | Time 70.2805
    # sampler memory (MB): 2323.9375
    # sampler comp (MB): 14644.671875
    # """

    # """
    # [10, 10, 10], amp_rate=2, refresh_rate=0.3, T=50
    # lstime.mean (s): 0.13441514268852062
    # lsmem.mean (MB): 17.331633203799655
    # Epoch 00002 | Loss 0.0000 | Time 27.1706
    # sampler memory (MB): 1471.875
    # sampler comp (MB): 11508.421875
    # """

    # """
    # [5, 5, 5], amp_rate=2, refresh_rate=0.3, T=50
    # lstime.mean (s): 0.06801044426227892
    # lsmem.mean (MB): 8.788752158894646
    # Epoch 00002 | Loss 0.0000 | Time 14.0671
    # sampler memory (MB): 2431.359375
    # sampler comp (MB): 7521.640625
    # """

    # """
    # [20, 20, 20], amp_rate=1.5, refresh_rate=0.4, T=50
    # lstime.mean (s): 0.35511266094003524
    # lsmem.mean (MB): 20.667746113989637
    # Epoch 00002 | Loss 0.0000 | Time 69.8808
    # sampler memory (MB): 3233.5
    # sampler comp (MB): 15201.65625
    # """

    # """
    # [10, 10, 10], amp_rate=1.5, refresh_rate=0.4, T=50
    # lstime.mean (s): 0.13539677374515302
    # lsmem.mean (MB): 16.850091753022454
    # Epoch 00002 | Loss 0.0000 | Time 27.2500
    # sampler memory (MB): 2216.90625
    # sampler comp (MB): 11974.625
    # """

    # """
    # [5, 5, 5], amp_rate=1.5, refresh_rate=0.4, T=50
    # lstime.mean (s): 0.06892447973985952
    # lsmem.mean (MB): 10.312041234887737
    # Epoch 00002 | Loss 0.0000 | Time 14.1827
    # sampler memory (MB): 1340.59375
    # sampler comp (MB): 7311.25
    # """

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