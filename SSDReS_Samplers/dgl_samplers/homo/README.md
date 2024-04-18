homo-graph acc & mem-reduction results

Method Explanation
```
NeighborSampler_FCR_struct: Fully Cache Refresh, with each hop has unique cached frontier
NeighborSampler_FCR_struct_shared_cache: Fully Cache Refresh with Shared Cache, with all hop has shared cached frontier
NeighborSampler_OTF_struct_FSCRFCF: Fully cache refresh + Fully cache fetch
NeighborSampler_OTF_struct_FSCRFCF_shared_cache: Fully cache refresh + Fully cache fetch with shared cache
NeighborSampler_OTF_struct_PCFFSCR: Fully cache refresh + Partially cache fetch
NeighborSampler_OTF_struct_PCFFSCR_shared_cache: Fully cache refresh + Partially cache fetch with shared cache
NeighborSampler_OTF_struct_PCFPSCR: Partially cache refresh + Partially cache fetch
NeighborSampler_OTF_struct_PCFPSCR_SC: Partially cache refresh + Partially cache fetch with shared cache
NeighborSampler_OTF_struct_PSCRFCF: Partially cache refresh + Fully cache fetch
NeighborSampler_OTF_struct_PSCRFCF_SC: Partially cache refresh + Fully cache fetch with shared cache
```

# FBL

### Experimental Settings

| IDs       | Experimental Settings                  |
|-----------|----------------------------------------|
| Setting 1 | fanouts=[20, 20, 20], batch_size = 128 |
| Setting 2 | fanouts=[10, 10, 10], batch_size = 128 |
| Setting 3 | fanouts=[5, 5, 5], batch_size = 128    |

### Impact on Memory and Training Time

| Metric           | Setting 1 | Setting 2 | Setting 3 |
|------------------|-----------|-----------|-----------|
| Loading Time (s) | 0.28      | 0.075     | 0.019     |
| Memory Usage (MB)| 6.34      | 4.70      | 4.60      |
| GPU Usage        | graph size| graph size| graph size|

### Impact on Accuracy

| Accuracy   | GCN    | GAT    | GraphSAGE |
|------------|--------|--------|-----------|
| Setting 1  | 0.7178 | 0.7050 | 0.7250    |
| Setting 2  | 0.7024 | 0.7067 | 0.7140    |
| Setting 3  | 0.7098 | 0.7123 | 0.7184    |

# FCR

### Experimental Settings

| IDs        | Experimental Settings                               |
|------------|-----------------------------------------------------|
| Setting 1  | fanouts=[20, 20, 20], alpha=2, T=50, batch_size=128 |
| Setting 2  | fanouts=[10, 10, 10], alpha=2, T=50, batch_size=128 |
| Setting 3  | fanouts=[5, 5, 5], alpha=2, T=50, batch_size=128    |
| Setting 4  | fanouts=[20, 20, 20], alpha=1.5, T=50, batch_size=128|
| Setting 5  | fanouts=[10, 10, 10], alpha=1.5, T=50, batch_size=128|
| Setting 6  | fanouts=[5, 5, 5], alpha=1.5, T=50, batch_size=128   |

### Impact on Memory and Training Time

| IDs        | Loading Time (s) | Memory Usage (MB) |  GPU Usage (MB)     |
|------------|------------------|-------------------|---------------------|
| Setting 1  | 0.2571           | 2.6962            | 5747.703            |
| Setting 2  | 0.0639           | 2.1149            | 4754.109            |
| Setting 3  | 0.0163           | 1.2921            | 3828.813            |
| Setting 4  | 0.2480           | 6.5989            | 5278.156            |
| Setting 5  | 0.0603           | 2.0253            | 4837.313            |
| Setting 6  | 0.0152           | 1.8717            | 3618.922            |

Note: GPU Usage is roughly equal to " α * fanout structure" (α represents alpha, and f represents the fanout structure).

### Estimated Impact on Accuracy

| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7100  | 0.6980  | 0.7180    |
| Setting 2  | 0.6950  | 0.6990  | 0.7080    |
| Setting 3  | 0.7030  | 0.7050  | 0.7130    |
| Setting 4  | 0.7120  | 0.7000  | 0.7200    |
| Setting 5  | 0.6980  | 0.7000  | 0.7100    |
| Setting 6  | 0.7050  | 0.7080  | 0.7150    |

# FCR - SC

### Experimental Settings

| IDs        | Experimental Settings                               |
|------------|-----------------------------------------------------|
| Setting 1  | fanouts=[20, 20, 20], alpha=2, T=50, batch_size=128 |
| Setting 2  | fanouts=[10, 10, 10], alpha=2, T=50, batch_size=128 |
| Setting 3  | fanouts=[5, 5, 5], alpha=2, T=50, batch_size=128    |
| Setting 4  | fanouts=[20, 20, 20], alpha=1.5, T=50, batch_size=128|
| Setting 5  | fanouts=[10, 10, 10], alpha=1.5, T=50, batch_size=128|
| Setting 6  | fanouts=[5, 5, 5], alpha=1.5, T=50, batch_size=128   |

### Impact on Memory and Training Time

| IDs        | Loading Time (s) | Memory Usage (MB) | GPU Usage (MB)  |
|------------|------------------|-------------------|-----------------|
| Setting 1  | 0.2554           | 4.4287            | 1386.828        |
| Setting 2  | 0.0640           | 4.6272            | 2744.125        |
| Setting 3  | 0.0161           | 1.6645            | 2455.906        |
| Setting 4  | 0.2405           | 6.7136            | 527.312         |
| Setting 5  | 0.0592           | 4.2753            | 362.125         |
| Setting 6  | 0.0149           | 0.1065            | 1721.359        |

### Estimated Impact on Accuracy

| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7105  | 0.6985  | 0.7185    |
| Setting 2  | 0.6948  | 0.6987  | 0.7078    |
| Setting 3  | 0.7032  | 0.7052  | 0.7131    |
| Setting 4  | 0.7122  | 0.7001  | 0.7201    |
| Setting 5  | 0.6978  | 0.6998  | 0.7098    |
| Setting 6  | 0.7052  | 0.7081  | 0.7152    |


### Conclusion
No significiant changes in the accuracy, has significant changes in mem reduction.


(visualization on sampling speed improvement of graphs) - overall
(visualization on training time of GNNs)
(visualization of memory reduction of GNNs)



## OGBN-Products

### FBL

#### FBL Experimental Settings

| IDs       | Experimental Settings                                    |
|-----------|----------------------------------------------------------|
| Setting 1 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4]   |
| Setting 2 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4]   |
| Setting 3 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4] |

#### FBL - Results

| Metric        | Setting 1 (GCN) | Setting 2 (SGC) | Setting 3 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------------|
| Accuracy      | 0.7178          | 0.7050          | 0.7250                |
| Training Time | 3.6946 s        | 2.8000 s        | 4.0000 s              |
| Memory Usage  | 0.3 GB          | 0.25 GB         | 0.35 GB               |

### FCR

#### FCR Experimental Settings

| IDs       | Experimental Settings                                                           |
|-----------|---------------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50            |
| Setting 2 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=2391          |
| Setting 3 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50            |
| Setting 4 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=2391          |
| Setting 5 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50      |
| Setting 6 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=2391    |

Note: T = 2391 is theoretical optimal

#### FCR - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7145          | 0.7071          | 0.7020          | 0.6950          | 0.7200                | 0.7150                |
| Training Time | 6.7892 s        | 3.4923 s        | 5.5000 s        | 3.2000 s        | 7.2000 s              | 4.0000 s              |
| Memory Usage  | 0.4 GB          | 0.5 GB          | 0.35 GB         | 0.45 GB         | 0.45 GB               | 0.55 GB               |

## Figure

## Conclusion
