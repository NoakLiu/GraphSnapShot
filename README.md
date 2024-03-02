### GraphSnapShot

GraphSnapShot is a framework for fast storage, retrieval and computation for graph learning, developed by Shawn Dong Liu. It can quickly store and update the local topology of graph structure, just like take `snapshots` of the graph.

![model construction](./assets/Disk_Cache_Memory_GraphSnapShot.png)

![model construction](./assets/SSDReS.png)3 system design strategies


```
FBL: full batch load
OTF: partial cache refresh (on the fly) snapshot
FCR: full cache refresh snapshot
```

Deployment:

FBL implementation is same as the `MultiLayerSampler` implemented in dgl.

To deploy our projects, we can reach to the Samplers in SSDReS_Sampler by `cd SSDReS_Samplers`, and then find the following file
```
NeighborSampler_OTF_struct.py
NeighborSampler_OTF_nodes.py
NeighborSampler_FCR_struct.py
NeighborSampler_FCR_nodes.py
```

Add samplers code in SSDReS_Sampler into the neighbor_sampler.py in dgl as in the path above and save the changes.

```
cd ~/anaconda3/envs/dglsampler/lib/python3.9/site-packages/dgl/dataloading/neighbor_sampler.py
```
Then you can deploy OTF and FCR samplers at node-level and struct-level from neighbor_sampler and create objects of those samplers.

FCR in execution

https://github.com/NoakLiu/GraphSnapShot/assets/116571268/ed701012-9267-4860-845b-baf1c39c317c

OTF in execution

https://github.com/NoakLiu/GraphSnapShot/assets/116571268/6fe1a566-d4e9-45ae-b654-676a2e4d6a58


Two types of samplers:
```
node-level: split graph into graph_static and graph_dynamic, enhance the capability for CPU-GPU co-utilization.
structure-level: reduce the inefficiency of resample whole k-hop structure for each node, use static-presample and dynamic-resample for structure retrieval acceleration.
```

Downsteam Task: 
```
MultiLayer GCN
MultiLayer SGC
MultiLayer GraphSAGE
```

Datasets:
```
ogbn_arxiv - node classification
ogbn_products - node classification
ogbn_proteins - node classification

```

<p align="center">
  <img src="./ogbn-arxiv_degree_distribution.png" width="200" />
  <img src="./ogbn-products_degree_distribution.png" width="200" />
  <img src="./ogbn-proteins_degree_distribution.png" width="200" />
</p>

Design of FBL

![model construction](./assets/FBL.png)

Design of OTF

![model construction](./assets/OTF.png)

Design of FCR

![model construction](./assets/FCR.png)



Results

OTF - Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=1
```
Epoch 00000 | Loss 1.6886 | Accuracy 0.5821 | Time 98.2462
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 26.79it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:38<00:00, 15.57it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:29<00:00, 20.30it/s]
Test accuracy 0.7021
```
OTF - Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50
```
Epoch 00000 | Loss 0.9100 | Accuracy 0.7954 | Time 3.9952
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:23<00:00, 25.64it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:40<00:00, 14.87it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:30<00:00, 19.84it/s]
Test accuracy 0.7267
```
OTF Shared Cache Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=1
```
Epoch 00000 | Loss 1.1858 | Accuracy 0.7172 | Time 39.8936
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 26.10it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:39<00:00, 15.28it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:30<00:00, 19.56it/s]
Test accuracy 0.6963
```
OTF Shared Cache Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50
```
Epoch 00000 | Loss 1.1994 | Accuracy 0.6723 | Time 2.4863
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 26.91it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:38<00:00, 15.64it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:29<00:00, 20.46it/s]
Test accuracy 0.7166
```
FCR Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, T=3
```
Epoch 00000 | Loss 0.8631 | Accuracy 0.8006 | Time 69.2330
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:20<00:00, 28.95it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:36<00:00, 16.37it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:26<00:00, 22.48it/s]
Test accuracy 0.7251
```
FCR Shared Cache Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, T=3
```
Epoch 00000 | Loss 0.9660 | Accuracy 0.7919 | Time 25.3724
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:21<00:00, 27.22it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:36<00:00, 16.31it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:28<00:00, 21.17it/s]
Test accuracy 0.7136
```
FCR Shared Cache Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, T=30
```
Epoch 00000 | Loss 1.0913 | Accuracy 0.7569 | Time 5.3640
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 26.70it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:40<00:00, 14.93it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:30<00:00, 19.35it/s]
Test accuracy 0.6947
```

FBL Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4]
```
Epoch 00000 | Loss 0.9333 | Accuracy 0.7865 | Time 3.6946
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 26.46it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:37<00:00, 15.91it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:27<00:00, 21.41it/s]
Test accuracy 0.7178
```

```
time = [252.6, 143.4, 107.8]  
memory = [20.3, 18.2, 22.5]  
accuracy = [71.3, 62.9, 67.5]  
posttrain_time = [0, 23.6, 82.5]  
```

![model construction](./assets/res.png)

The main idea of this framework is `Static Pre-Sampling and Dynamic Re-Sampling` for Efficient Graph Learning Storage and Retrieval.

Result Analysis

setting 1:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 1
mode = "tradeoff"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_1hop_tradeoff/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_1hop_tradeoff/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_1hop_tradeoff/Training_Time_versus_Alpha_Changes.png)

setting 2.1:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 2
mode = "sswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_2hop_sswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_2hop_sswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_2hop_sswp/Training_Time_versus_Alpha_Changes.png)

setting 2.2:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 2
mode = "dswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_2hop_dswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_2hop_dswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_2hop_dswp/Training_Time_versus_Alpha_Changes.png)

setting 2.3:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 2
mode = "cswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_2hop_cswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_2hop_cswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_2hop_cswp/Training_Time_versus_Alpha_Changes.png)

setting 3.1:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 3
mode = "sswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_3hop_sswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_3hop_sswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_3hop_sswp/Training_Time_versus_Alpha_Changes.png)

setting 3.2:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 3
mode = "dswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_3hop_dswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_3hop_dswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_3hop_dswp/Training_Time_versus_Alpha_Changes.png)

setting 3.3:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 3
mode = "cswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_3hop_cswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_3hop_cswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_3hop_cswp/Training_Time_versus_Alpha_Changes.png)



setting 3.1:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 4
mode = "sswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_4hop_sswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_4hop_sswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_4hop_sswp/Training_Time_versus_Alpha_Changes.png)

setting 3.2:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 4
mode = "dswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_4hop_dswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_4hop_dswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_4hop_dswp/Training_Time_versus_Alpha_Changes.png)

setting 3.3:

```Python
presampled_nodes = 20
presampled_perexpansion = 2
resampled_nodes = 10
sampled_depth = 4
mode = "cswp"
```



1. Test Loss versus Alpha Changes

   ![Test Loss versus Alpha Changes](results_4hop_cswp/Test_Loss_versus_Alpha_Changes.png)

2. Test Accuracy versus Alpha Changes

   ![Test Accuracy versus Alpha Changes](results_4hop_cswp/Test_Accuracy_versus_Alpha_Changes.png)

3. Training Time versus Alpha Changes

   ![Training Time versus Alpha Changes](results_4hop_cswp/Training_Time_versus_Alpha_Changes.png)