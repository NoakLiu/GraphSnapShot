#### Sparse & Dense Processing for GPU Usage Reducation
- For homograph
    - 1. python div_graph_by_deg_homo.py --> dense graph, sparse graph
    - 2. deploy homo SSDReS samplers such as FCR, FCR-SC, OTF((PR, FR)x(PF, FF)), OTF((PR, FR)x(PF, FF))-SC on dense graph
    - 3. deploy FBL on sparse graph

- For hetegraph
    - 1. python div_graph_by_deg_hete.py --> dense graph, sparse graph
    - 2. deploy homo SSDReS samplers such as FCR_hete, FCR-SC_hete, OTF((PR, FR)x(PF, FF))_hete, OTF((PR, FR)x(PF, FF))-SC_hete on dense graph
    - 3. deploy FBL on sparse graph

## GPU Usage Figure
![model construction](/results/hete/gpu_by_thrs_mag.png)

![model construction](/results/homo/gpu_by_thrs_arxiv.png)

![model construction](/results/homo/gpu_by_thrs_products.png)

## Analysis
The plots illustrate the GPU usage in terms of the number of edges for three datasets (ogbn-arxiv, ogbn-products, and ogbn-mag) at various node degree thresholds. They show the efficiency of handling dense and sparse parts of graphs separately while optimizing GPU usage.

In the provided approach, for the sparse regions of the graph, all edges are fully loaded into the cache, as these regions are less dense and involve fewer edges, making them ideal for full caching. This allows for rapid access to the sparse regions during processing without repeatedly fetching data from memory, thereby reducing memory bandwidth and latency.

On the other hand, the dense regions of the graph, which could be prohibitively large and may not fit entirely in the cache, are handled using GraphSnapShot's sampling samplers for efficient sampling, retrieval and storage. This allows for a representative subset of the dense regions to be processed, significantly reducing the amount of GPU memory required.

The advantages of this approach are evident in the plots, where the total GPU usage (green line) is minimized at a specific threshold. At this threshold, the balance between fully cached sparse edges and sampled dense edges leads to the lowest total number of edges being processed by the GPU. The red dots on the plots represent this optimal balance point, showcasing the lowest GPU usage across the tested thresholds.

By employing the GraphSnapShot technique, we leverage the natural structure of real-world graphs, which often follow a power-law distribution where the majority of nodes have low degrees (sparse regions), and a minority of nodes have high degrees (dense regions). This strategy optimally manages memory and computational resources:

Firstly, it shows memory efficiency, as sparse regions require less memory, so caching them entirely is feasible. Dense regions, when sampled, reduce the overall memory footprint.
Secondly, it shows computation Efficiency, as processing the full set of edges in sparse regions is quick due to their low density, while sampling in dense regions reduces the number of computations required.

Overall, the sparse & dense split processing methodology in GraphSnapShot demonstrates a pragmatic approach to handling large-scale graphs by optimizing GPU resource utilization, balancing between full data retention for sparse parts and intelligent sampling for dense parts, thereby enhancing performance and efficiency.