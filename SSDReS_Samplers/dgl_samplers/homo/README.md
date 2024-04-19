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

graph size:

products = 1454.859375 MB
arxiv = 404.40625 MB
mag = 1867.578125 MB

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

### Impact on Accuracy

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

### Impact on Accuracy

| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7105  | 0.6985  | 0.7185    |
| Setting 2  | 0.6948  | 0.6987  | 0.7078    |
| Setting 3  | 0.7032  | 0.7052  | 0.7131    |
| Setting 4  | 0.7122  | 0.7001  | 0.7201    |
| Setting 5  | 0.6978  | 0.6998  | 0.7098    |
| Setting 6  | 0.7052  | 0.7081  | 0.7152    |

# OTF (FSCRFCF mode)
### Experimental Settings

| IDs        | Experimental Settings                                           |
|------------|-----------------------------------------------------------------|
| Setting 1  | fanouts=[20, 20, 20], amp_rate=2, refresh_rate=0.15, T=358      |
| Setting 2  | fanouts=[20, 20, 20], amp_rate=2, refresh_rate=0.15, T=50       |
| Setting 3  | fanouts=[10, 10, 10], amp_rate=2, refresh_rate=0.15, T=50       |
| Setting 4  | fanouts=[5, 5, 5], amp_rate=2, refresh_rate=0.15, T=50          |
| Setting 5  | fanouts=[20, 20, 20], amp_rate=1.5, refresh_rate=0.15, T=50     |
| Setting 6  | fanouts=[10, 10, 10], amp_rate=1.5, refresh_rate=0.15, T=50     |
| Setting 7  | fanouts=[5, 5, 5], amp_rate=1.5, refresh_rate=0.15, T=50        |
| Setting 8  | fanouts=[20, 20, 20], amp_rate=1.5, refresh_rate=0.3, T=50      |
| Setting 9  | fanouts=[10, 10, 10], amp_rate=1.5, refresh_rate=0.3, T=50      |
| Setting 10 | fanouts=[5, 5, 5], amp_rate=1.5, refresh_rate=0.3, T=50         |

### Impact on Memory and Training Time
| IDs        | Loading Time (s) | Memory Usage (MB) | GPU Usage (MB) |
|------------|------------------|-------------------|---------------------|
| Setting 1  | 0.2460           | 4.1327            | 2657.891            |
| Setting 2  | 0.2528           | 4.1924            | 2924.234            |
| Setting 3  | 0.0568           | 1.8782            | 1020.250            |
| Setting 4  | 0.0145           | 0.3207            | 1240.438            |
| Setting 5  | 0.2425           | 2.5888            | 2510.609            |
| Setting 6  | 0.0556           | 2.7754            | 1660.063            |
| Setting 7  | 0.0141           | -0.6624           | 794.359             |
| Setting 8  | 0.2495           | 3.8386            | 2818.844            |
| Setting 9  | 0.0558           | 1.5645            | 1655.469            |
| Setting 10 | 0.0146           | -0.3279           | 1102.094            |

### Impact on Accuracy

| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7090  | 0.6950  | 0.7170    |
| Setting 2  | 0.7100  | 0.6980  | 0.7190    |
| Setting 3  | 0.7010  | 0.7020  | 0.7110    |
| Setting 4  | 0.7000  | 0.7030  | 0.7120    |
| Setting 5  | 0.7080  | 0.6940  | 0.7160    |
| Setting 6  | 0.6990  | 0.6970  | 0.7100    |
| Setting 7  | 0.6980  | 0.6960  | 0.7090    |
| Setting 8  | 0.7070  | 0.6930  | 0.7150    |
| Setting 9  | 0.6980  | 0.6950  | 0.7080    |
| Setting 10 | 0.6970  | 0.6940  | 0.7070    |

# OTF (FSCR FCF)

### Experimental Settings
| IDs        | Experimental Settings                                     |
|------------|-----------------------------------------------------------|
| Setting 1  | fanouts=[20, 20, 20], amp_rate=1.5, fetch_rate=0.4, T_fetch=10 |
| Setting 2  | fanouts=[10, 10, 10], amp_rate=1.5, fetch_rate=0.4, T_fetch=10 |
| Setting 3  | fanouts=[5, 5, 5], amp_rate=1.5, fetch_rate=0.4, T_fetch=10    |
| Setting 4  | fanouts=[20, 20, 20], amp_rate=2, fetch_rate=0.3, T_fetch=10   |
| Setting 5  | fanouts=[10, 10, 10], amp_rate=2, fetch_rate=0.3, T_fetch=10   |
| Setting 6  | fanouts=[5, 5, 5], amp_rate=2, fetch_rate=0.3, T_fetch=10      |

### Impact on Memory and Training Time
| IDs        | Loading Time (s) | Memory Usage (MB) | GPU Usage (MB)  |
|------------|------------------|-------------------|-----------------|
| Setting 1  | 0.3890           | -8.9206           | 2604.016        |
| Setting 2  | 0.1056           | 2.5692            | 1727.594        |
| Setting 3  | 0.0644           | 0.8608            | 1090.953        |
| Setting 4  | 0.2892           | 1.3425            | 4554.438        |
| Setting 5  | 0.1104           | 2.9305            | 3284.250        |
| Setting 6  | 0.0649           | 1.1805            | 1313.563        |

### Impact on Accuracy
| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7100  | 0.6970  | 0.7190    |
| Setting 2  | 0.7060  | 0.6940  | 0.7150    |
| Setting 3  | 0.7030  | 0.6920  | 0.7110    |
| Setting 4  | 0.7120  | 0.7000  | 0.7220    |
| Setting 5  | 0.7090  | 0.6970  | 0.7180    |
| Setting 6  | 0.7050  | 0.6930  | 0.7140    |

# OTF (FSCRFCF mode - shared cache)
### Experimental Settings

| IDs        | Experimental Settings                                       |
|------------|-------------------------------------------------------------|
| Setting 1  | fanouts=[20, 20, 20], amp_rate=2, refresh_rate=0.15, T=50   |
| Setting 2  | fanouts=[10, 10, 10], amp_rate=2, refresh_rate=0.15, T=50   |
| Setting 3  | fanouts=[5, 5, 5], amp_rate=2, refresh_rate=0.15, T=50      |
| Setting 4  | fanouts=[20, 20, 20], amp_rate=1.5, refresh_rate=0.15, T=50 |
| Setting 5  | fanouts=[10, 10, 10], amp_rate=1.5, refresh_rate=0.15, T=50 |
| Setting 6  | fanouts=[5, 5, 5], amp_rate=1.5, refresh_rate=0.15, T=50    |
| Setting 7  | fanouts=[20, 20, 20], amp_rate=2, refresh_rate=0.3, T=50    |
| Setting 8  | fanouts=[10, 10, 10], amp_rate=2, refresh_rate=0.3, T=50    |
| Setting 9  | fanouts=[5, 5, 5], amp_rate=2, refresh_rate=0.3, T=50       |

### Impact on Memory and Training Time

| IDs        | Loading Time (s) | Memory Usage (MB) | GPU Usage (MB) |
|------------|------------------|-------------------|---------------------|
| Setting 1  | 0.2393           | 0.6762            | 2423.484            |
| Setting 2  | 0.0559           | 0.8663            | 2064.453            |
| Setting 3  | 0.0133           | 1.4149            | 176.578             |
| Setting 4  | 0.2326           | 0.5645            | 2316.125            |
| Setting 5  | 0.0527           | 2.6571            | 252.422             |
| Setting 6  | 0.0126           | 0.4073            | 131.453             |
| Setting 7  | 0.2390           | 3.6368            | 967.938             |
| Setting 8  | 0.0560           | 2.7602            | 447.375             |
| Setting 9  | 0.0133           | 0.0318            | 1819.391            |

### Impact on Accuracy

| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7090  | 0.6950  | 0.7170    |
| Setting 2  | 0.7100  | 0.6980  | 0.7190    |
| Setting 3  | 0.7010  | 0.7020  | 0.7110    |
| Setting 4  | 0.7000  | 0.7030  | 0.7120    |
| Setting 5  | 0.7080  | 0.6940  | 0.7160    |
| Setting 6  | 0.6990  | 0.6970  | 0.7100    |
| Setting 7  | 0.6980  | 0.6960  | 0.7090    |
| Setting 8  | 0.7070  | 0.6930  | 0.7150    |
| Setting 9  | 0.6980  | 0.6950  | 0.7080    |
| Setting 10 | 0.6970  | 0.6940  | 0.7070    |

# OTF (PSCR FCF mode)
### Experimental Settings
| IDs        | Experimental Settings                                      |
|------------|------------------------------------------------------------|
| Setting 1  | fanouts=[20, 20, 20], amp_rate=2, refresh_rate=0.3, T=50   |
| Setting 2  | fanouts=[10, 10, 10], amp_rate=2, refresh_rate=0.3, T=50   |
| Setting 3  | fanouts=[5, 5, 5], amp_rate=2, refresh_rate=0.3, T=50      |
| Setting 4  | fanouts=[20, 20, 20], amp_rate=1.5, refresh_rate=0.4, T=50 |
| Setting 5  | fanouts=[10, 10, 10], amp_rate=1.5, refresh_rate=0.4, T=50 |
| Setting 6  | fanouts=[5, 5, 5], amp_rate=1.5, refresh_rate=0.4, T=50    |

### Impact on Memory and Training Time
| IDs        | Loading Time (s) | Memory Usage (MB) | GPU Usage (MB)  |
|------------|------------------|-------------------|-----------------|
| Setting 1  | 0.3558           | 21.2783           | 2323.938        |
| Setting 2  | 0.1344           | 17.3316           | 1471.875        |
| Setting 3  | 0.0680           | 8.7888            | 2431.359        |
| Setting 4  | 0.3551           | 20.6677           | 3233.500        |
| Setting 5  | 0.1354           | 16.8501           | 2216.906        |
| Setting 6  | 0.0689           | 10.3120           | 1340.594        |

### Impact on Accuracy
| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7120  | 0.6990  | 0.7210    |
| Setting 2  | 0.7080  | 0.6960  | 0.7170    |
| Setting 3  | 0.7050  | 0.6930  | 0.7140    |
| Setting 4  | 0.7130  | 0.7000  | 0.7220    |
| Setting 5  | 0.7100  | 0.6980  | 0.7190    |
| Setting 6  | 0.7060  | 0.6940  | 0.7150    |


# OTF (PCF PSCR SC mode)

| IDs        | Experimental Settings                                      |
|------------|------------------------------------------------------------|
| Setting 1  | fanouts=[20, 20, 20], amp_rate=1.5, fetch_rate=0.4, T_fetch=10 |
| Setting 2  | fanouts=[10, 10, 10], amp_rate=1.5, fetch_rate=0.4, T_fetch=10 |
| Setting 3  | fanouts=[5, 5, 5], amp_rate=1.5, fetch_rate=0.4, T_fetch=10    |
| Setting 4  | fanouts=[20, 20, 20], amp_rate=2, fetch_rate=0.3, T_fetch=10   |
| Setting 5  | fanouts=[10, 10, 10], amp_rate=2, fetch_rate=0.3, T_fetch=10   |
| Setting 6  | fanouts=[5, 5, 5], amp_rate=2, fetch_rate=0.3, T_fetch=10      |

Impact

| IDs        | Loading Time (s) | Memory Usage (MB) | GPU Usage (MB)  |
|------------|------------------|-------------------|-----------------|
| Setting 1  | 0.6797           | 14.8317           | 3283.891        |
| Setting 2  | 0.3963           | 12.6763           | 1304.078        |
| Setting 3  | 0.2564           | 5.4109            | 1741.734        |
| Setting 4  | 0.8273           | 15.3318           | 2477.234        |
| Setting 5  | 0.4560           | 16.6057           | 497.906         |
| Setting 6  | 0.2931           | 12.7773           | 263.516         |

Accuracy
| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7120  | 0.6990  | 0.7210    |
| Setting 2  | 0.7080  | 0.6960  | 0.7170    |
| Setting 3  | 0.7050  | 0.6930  | 0.7140    |
| Setting 4  | 0.7130  | 0.7000  | 0.7220    |
| Setting 5  | 0.7100  | 0.6970  | 0.7190    |
| Setting 6  | 0.7060  | 0.6940  | 0.7150    |

# OTF (PSCR FCF SC mode)
### Experimental Settings
| IDs        | Experimental Settings                                        |
|------------|--------------------------------------------------------------|
| Setting 1  | fanouts=[20, 20, 20], amp_rate=1.5, fetch_rate=0.4, T_fetch=50 |
| Setting 2  | fanouts=[10, 10, 10], amp_rate=1.5, fetch_rate=0.4, T_fetch=50 |
| Setting 3  | fanouts=[5, 5, 5], amp_rate=1.5, fetch_rate=0.4, T_fetch=50    |
| Setting 4  | fanouts=[20, 20, 20], amp_rate=2, fetch_rate=0.3, T_fetch=50   |
| Setting 5  | fanouts=[10, 10, 10], amp_rate=2, fetch_rate=0.3, T_fetch=50   |
| Setting 6  | fanouts=[5, 5, 5], amp_rate=2, fetch_rate=0.3, T_fetch=50      |

### Impact on Memory and Training Time
| IDs        | Loading Time (s) | Memory Usage (MB) | GPU Usage (MB)  |
|------------|------------------|-------------------|-----------------|
| Setting 1  | 0.3202           | 14.0887           | 1571.062        |
| Setting 2  | 0.1214           | 10.0932           | 1962.359        |
| Setting 3  | 0.0585           | 4.3968            | 1220.250        |
| Setting 4  | 0.3521           | 15.5863           | 1526.328        |
| Setting 5  | 0.1309           | 13.4031           | 1134.344        |
| Setting 6  | 0.0672           | 7.4897            | 867.438         |

### Impact on Accuracy
| IDs        | GCN     | GAT     | GraphSAGE |
|------------|---------|---------|-----------|
| Setting 1  | 0.7110  | 0.6980  | 0.7200    |
| Setting 2  | 0.7070  | 0.6950  | 0.7160    |
| Setting 3  | 0.7040  | 0.6920  | 0.7130    |
| Setting 4  | 0.7130  | 0.7010  | 0.7220    |
| Setting 5  | 0.7100  | 0.6980  | 0.7190    |
| Setting 6  | 0.7060  | 0.6940  | 0.7150    |




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

## Conclusion
