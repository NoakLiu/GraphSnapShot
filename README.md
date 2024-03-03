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

| Feature            | OGBN-ARXIV      | OGBN-PRODUCTS   | OGBN-PROTEINS   |
|--------------------|-----------------|-----------------|-----------------|
| Dataset Type       | Citation Network| Product Purchase| Protein-Protein |
| Number of Nodes    | Approx. 17,700  | Approx. 24,000  | Approx. 13,000  |
| Number of Edges    | Approx. 200,000 | Approx. 140,000 | Approx. 70,000  |
| Feature Dimension  | 128             | 100             | 50              |
| Number of Classes  | 40              | 89              | 112             |
| Number of Train Nodes | Approx. 10,000 | Approx. 12,000 | Approx. 6,000   |
| Number of Validation Nodes | Approx. 3,000 | Approx. 4,000 | Approx. 2,000 |
| Number of Test Nodes | Approx. 4,000 | Approx. 8,000   | Approx. 5,000   |
| Supervised Task    | Node Classification | Node Classification | Node Classification |
| Data Collection Year | 2018          | 2018            | 2018            |

Design of FBL

![model construction](./assets/FBL.png)

Design of OTF

![model construction](./assets/OTF.png)

Design of FCR

![model construction](./assets/FCR.png)



Results

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
OTF - Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 (theoretical optimal)
```
Epoch 00000 | Loss 1.0040 | Accuracy 0.7597 | Time 3.9875
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:23<00:00, 25.47it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:40<00:00, 14.83it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:29<00:00, 20.16it/s]
Test accuracy 0.7122
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
OTF Shared Cache Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 (theoretical optimal)
```
Epoch 00000 | Loss 1.1047 | Accuracy 0.6662 | Time 2.7414
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 26.46it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:38<00:00, 15.43it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:29<00:00, 19.94it/s]
Test accuracy 0.7096
```

FCR Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, T=50
```
Epoch 00000 | Loss 0.9367 | Accuracy 0.8043 | Time 6.7892
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 26.81it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:39<00:00, 15.22it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:30<00:00, 19.87it/s]
Test accuracy 0.7145
```
FCR Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, T=2391 (theoretical optimal)
```
Epoch 00000 | Loss 1.0370 | Accuracy 0.7860 | Time 3.4923
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:21<00:00, 27.29it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:39<00:00, 15.18it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:29<00:00, 20.07it/s]
Test accuracy 0.7071
```
FCR Shared Cache Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, T=50
```
Epoch 00000 | Loss 0.9180 | Accuracy 0.7974 | Time 4.2614
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 27.05it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:39<00:00, 15.33it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:29<00:00, 20.34it/s]
Test accuracy 0.7265
```
FCR Shared Cache Setting: dataset: ogbn-products, sampler1: [2,2,2] sampler2: [4,4,4], alpha=2, T=797 (theoretical optimal)
```
Epoch 00000 | Loss 1.0245 | Accuracy 0.7795 | Time 3.2381
Testing...
ModuleList(
  (0): GraphConv(in=100, out=256, normalization=both, activation=None)
  (1): GraphConv(in=256, out=256, normalization=both, activation=None)
  (2): GraphConv(in=256, out=47, normalization=both, activation=None)
)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:22<00:00, 26.55it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:39<00:00, 15.21it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 598/598 [00:29<00:00, 20.02it/s]
Test accuracy 0.7207
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