Dense Graph GraphSnapShot Cache for SSDReS_Samplers

## Method
- For sparse graphs, FBL method will be directedly deployed
- For dense graphs, SSDReS methods will be deployed

## SSDReS Samplers
- dgl samplers
    - hete
        - FCR_hete
        - FCR_SC_hete
        - OTF((PR, FR)x(PF, FF))_hete
        - OTF((PR, FR)x(PF, FF))_SC_hete
    - homo
        - FCR
        - FCR_SC
        - OTF((PR, FR)x(PF, FF))
        - OTF((PR, FR)x(PF, FF))_SC

- dgl samplers simple
    - hete
        - FCR_hete
        - FCR_SC_hete
        - OTF_hete
        - OTF_SC_hete
    - homo
        - FCR
        - FCR_SC
        - OTF
        - OTF_SC

## Deployment Sequence
- For homograph
    - 1. python div_graph_by_deg_homo.py --> dense graph, sparse graph
    - 2. deploy homo SSDReS samplers such as FCR, FCR-SC, OTF((PR, FR)x(PF, FF)), OTF((PR, FR)x(PF, FF))-SC on dense graph
    - 3. deploy FBL on sparse graph

- For hetegraph
    - 1. python div_graph_by_deg_hete.py --> dense graph, sparse graph
    - 2. deploy homo SSDReS samplers such as FCR_hete, FCR-SC_hete, OTF((PR, FR)x(PF, FF))_hete, OTF((PR, FR)x(PF, FF))-SC_hete on dense graph
    - 3. deploy FBL on sparse graph

## Figure
![model construction](/assets/dense_proc.png)

## Deployment Sequence
For homograph
    1. python div_graph_by_deg_homo.py --> dense graph, sparse graph
    2. deploy homo SSDReS samplers such as FCR, FCR-SC, OTF((PR, FR)x(PF, FF)), OTF((PR, FR)x(PF, FF))-SC on dense graph
    3. deploy FBL on sparse graph

For hetegraph
    1. python div_graph_by_deg_hete.py --> dense graph, sparse graph
    2. deploy homo SSDReS samplers such as FCR_hete, FCR-SC_hete, OTF((PR, FR)x(PF, FF))_hete, OTF((PR, FR)x(PF, FF))-SC_hete on dense graph
    3. deploy FBL on sparse graph

## Figures for memory reduction

![homo mem reduction](/results/homo/sample_efficiency_homo_arxiv.png)

![homo mem reduction](/results/homo/sample_efficiency_homo_products.png)

![hete mem reduction](/results/hete/sample_efficiency_hete_mag.png)

## Analysis
The key point of GraphSnapShot is to cache the local structure instead of whole graph input for memory reduction and sampling efficiency.

## OGBN-ARXIV

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

### FCR - Shared Cache 
#### Experimental Settings

| IDs       | Experimental Settings                                                  |
|-----------|------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50  |
| Setting 2 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |
| Setting 3 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50  |
| Setting 4 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |
| Setting 5 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50 |
| Setting 6 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |

Note: T = 797 is theoretical optimal

#### FCR - Shared Cache - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7145          | 0.7071          | 0.7020          | 0.6950          | 0.7200                | 0.7150                |
| Training Time | 6.7892 s        | 3.4923 s        | 5.5000 s        | 3.2000 s        | 7.2000 s              | 4.0000 s              |
| Memory Usage  | 0.4 GB          | 0.5 GB          | 0.35 GB         | 0.45 GB         | 0.45 GB               | 0.55 GB               |


### OTF 

#### Experimental Settings

| IDs       | Experimental Settings                                                                 |
|-----------|---------------------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 2 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |
| Setting 3 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 4 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |
| Setting 5 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 6 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |

Note: T = 358 is theoretical optimal

#### OTF - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7122          | 0.7178          | 0.7000          | 0.7050          | 0.7180                | 0.7220                |
| Training Time | 3.9952 s        | 3.9875 s        | 3.2000 s        | 3.1000 s        | 4.2000 s              | 4.0000 s              |
| Memory Usage  | 0.9 GB          | 0.4 GB          | 0.8 GB          | 0.35 GB         | 1.0 GB                | 0.45 GB               |


### OTF - Shared Cache 

#### Experimental Settings

| IDs       | Experimental Settings                                                                 |
|-----------|---------------------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 2 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |
| Setting 3 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 4 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |
| Setting 5 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 6 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |

Note: T=119 is theoretical optimal

#### OTF - Shared Cache - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7122          | 0.7178          | 0.7000          | 0.7050          | 0.7180                | 0.7220                |
| Training Time | 3.9952 s        | 3.9875 s        | 3.2000 s        | 3.1000 s        | 4.2000 s              | 4.0000 s              |
| Memory Usage  | 0.9 GB          | 0.4 GB          | 0.8 GB          | 0.35 GB         | 1.0 GB                | 0.45 GB               |

## OGBN-PRODUCTS

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

### FCR - Shared Cache 
#### Experimental Settings

| IDs       | Experimental Settings                                                  |
|-----------|------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50  |
| Setting 2 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |
| Setting 3 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50  |
| Setting 4 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |
| Setting 5 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50 |
| Setting 6 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |

Note: T = 797 is theoretical optimal

#### FCR - Shared Cache - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7145          | 0.7071          | 0.7020          | 0.6950          | 0.7200                | 0.7150                |
| Training Time | 6.7892 s        | 3.4923 s        | 5.5000 s        | 3.2000 s        | 7.2000 s              | 4.0000 s              |
| Memory Usage  | 0.4 GB          | 0.5 GB          | 0.35 GB         | 0.45 GB         | 0.45 GB               | 0.55 GB               |


### OTF 

#### Experimental Settings

| IDs       | Experimental Settings                                                                 |
|-----------|---------------------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 2 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |
| Setting 3 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 4 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |
| Setting 5 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 6 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |

Note: T = 358 is theoretical optimal

#### OTF - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7122          | 0.7178          | 0.7000          | 0.7050          | 0.7180                | 0.7220                |
| Training Time | 3.9952 s        | 3.9875 s        | 3.2000 s        | 3.1000 s        | 4.2000 s              | 4.0000 s              |
| Memory Usage  | 0.9 GB          | 0.4 GB          | 0.8 GB          | 0.35 GB         | 1.0 GB                | 0.45 GB               |


### OTF - Shared Cache 

#### Experimental Settings

| IDs       | Experimental Settings                                                                 |
|-----------|---------------------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 2 | Model: K-Hop GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |
| Setting 3 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 4 | Model: K-Hop SGC, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |
| Setting 5 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 6 | Model: K-Hop GraphSAGE, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |

Note: T=119 is theoretical optimal

#### OTF - Shared Cache - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7122          | 0.7178          | 0.7000          | 0.7050          | 0.7180                | 0.7220                |
| Training Time | 3.9952 s        | 3.9875 s        | 3.2000 s        | 3.1000 s        | 4.2000 s              | 4.0000 s              |
| Memory Usage  | 0.9 GB          | 0.4 GB          | 0.8 GB          | 0.35 GB         | 1.0 GB                | 0.45 GB               |

## OGBN-PROTEINS

### FBL

#### FBL Experimental Settings

| IDs       | Experimental Settings                                    |
|-----------|----------------------------------------------------------|
| Setting 1 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4]   |
| Setting 2 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4]   |
| Setting 3 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4] |

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
| Setting 1 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50            |
| Setting 2 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=2391          |
| Setting 3 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50            |
| Setting 4 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=2391          |
| Setting 5 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50      |
| Setting 6 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=2391    |

Note: T = 2391 is theoretical optimal

#### FCR - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7145          | 0.7071          | 0.7020          | 0.6950          | 0.7200                | 0.7150                |
| Training Time | 6.7892 s        | 3.4923 s        | 5.5000 s        | 3.2000 s        | 7.2000 s              | 4.0000 s              |
| Memory Usage  | 0.4 GB          | 0.5 GB          | 0.35 GB         | 0.45 GB         | 0.45 GB               | 0.55 GB               |

### FCR - Shared Cache 
#### Experimental Settings

| IDs       | Experimental Settings                                                  |
|-----------|------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50  |
| Setting 2 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |
| Setting 3 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50  |
| Setting 4 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |
| Setting 5 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=50 |
| Setting 6 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, T=797 |

Note: T = 797 is theoretical optimal

#### FCR - Shared Cache - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7145          | 0.7071          | 0.7020          | 0.6950          | 0.7200                | 0.7150                |
| Training Time | 6.7892 s        | 3.4923 s        | 5.5000 s        | 3.2000 s        | 7.2000 s              | 4.0000 s              |
| Memory Usage  | 0.4 GB          | 0.5 GB          | 0.35 GB         | 0.45 GB         | 0.45 GB               | 0.55 GB               |


### OTF 

#### Experimental Settings

| IDs       | Experimental Settings                                                                 |
|-----------|---------------------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 2 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |
| Setting 3 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 4 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |
| Setting 5 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 6 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=358 |

Note: T = 358 is theoretical optimal

#### OTF - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7122          | 0.7178          | 0.7000          | 0.7050          | 0.7180                | 0.7220                |
| Training Time | 3.9952 s        | 3.9875 s        | 3.2000 s        | 3.1000 s        | 4.2000 s              | 4.0000 s              |
| Memory Usage  | 0.9 GB          | 0.4 GB          | 0.8 GB          | 0.35 GB         | 1.0 GB                | 0.45 GB               |


### OTF - Shared Cache 

#### Experimental Settings

| IDs       | Experimental Settings                                                                 |
|-----------|---------------------------------------------------------------------------------------|
| Setting 1 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 2 | Model: K-Hop MWE-GCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |
| Setting 3 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 4 | Model: K-Hop MWE-DGCN, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |
| Setting 5 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=50 |
| Setting 6 | Model: K-Hop GAT, sampler1: [2,2,2], sampler2: [4,4,4], alpha=2, beta=2, gamma=0.15, T=119 |

Note: T=119 is theoretical optimal

#### OTF - Shared Cache - Results

| Metric        | Setting 1 (GCN) | Setting 2 (GCN) | Setting 3 (SGC) | Setting 4 (SGC) | Setting 5 (GraphSAGE) | Setting 6 (GraphSAGE) |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------------|-----------------------|
| Accuracy      | 0.7122          | 0.7178          | 0.7000          | 0.7050          | 0.7180                | 0.7220                |
| Training Time | 3.9952 s        | 3.9875 s        | 3.2000 s        | 3.1000 s        | 4.2000 s              | 4.0000 s              |
| Memory Usage  | 0.9 GB          | 0.4 GB          | 0.8 GB          | 0.35 GB         | 1.0 GB                | 0.45 GB               |


```
time = [252.6, 143.4, 107.8]  
memory = [20.3, 18.2, 22.5]  
accuracy = [71.3, 62.9, 67.5]  
posttrain_time = [0, 23.6, 82.5]  
```

![model construction](assets/res.png)

The main idea of this framework is `Static Pre-Sampling and Dynamic Re-Sampling` for Efficient Graph Learning Storage and Retrieval.
