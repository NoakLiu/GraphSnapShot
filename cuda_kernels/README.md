# GraphSnapShot CUDA Kernels

This directory contains the core CUDA kernels for GraphSnapShot, implementing high-performance graph sampling, caching, and memory management operations for the FBL, OTF, and FCR sampling strategies.

## Overview

GraphSnapShot CUDA kernels provide GPU-accelerated implementations of:

1. **Neighbor Sampling (FCR)**: Full Cache Refresh neighbor sampling with cache amplification
2. **Cache Refresh (OTF)**: On-The-Fly partial cache refresh for dynamic sampling
3. **Graph Structure Masking**: Dense graph filtering and memory reduction
4. **Buffer Management**: LRU cache management and memory optimization
5. **Multi-hop Aggregation**: Efficient k-hop neighbor sampling
6. **Weighted Edge Sampling**: Probability-based neighbor selection
7. **Memory Usage Detection**: Dynamic memory monitoring and optimization
8. **Heterogeneous Graph Sampling**: Multi-type node and edge sampling

## File Structure

```
cuda_kernels/
├── graphsnapshot_kernels.cu      # Main CUDA kernel implementations
├── graphsnapshot_kernels.h       # Header file with function declarations
├── python_bindings.cpp          # Python bindings using pybind11
├── cpu_fallback.cpp             # CPU fallback implementations
├── setup.py                     # Build configuration
└── README.md                    # This file
```

## Core Kernels

### 1. Neighbor Sampling FCR (`neighbor_sampling_fcr_kernel`)

Implements the Full Cache Refresh strategy with cache amplification:

```cuda
__global__ void neighbor_sampling_fcr_kernel(
    const int* seed_nodes,           // Input seed nodes
    const int num_seeds,             // Number of seed nodes
    const GraphData graph,           // Graph structure
    const CacheData cache,           // Cache data
    const int* fanouts,              // Fanout for each seed
    const float alpha,               // Cache amplification factor
    const bool use_cache,            // Whether to use cache
    int* sampled_neighbors,          // Output sampled neighbors
    int* neighbor_counts,            // Output neighbor counts
    curandState* states              // Random number states
);
```

**Key Features:**
- Cache amplification with alpha factor
- Efficient random sampling using curand
- Atomic operations for thread-safe neighbor counting
- Support for both cached and fresh sampling

### 2. Cache Refresh OTF (`cache_refresh_otf_kernel`)

Implements On-The-Fly cache refresh for dynamic sampling:

```cuda
__global__ void cache_refresh_otf_kernel(
    const int* seed_nodes,           // Input seed nodes
    const int num_seeds,             // Number of seed nodes
    const GraphData graph,           // Graph structure
    CacheData cache,                 // Cache data (modified)
    const float refresh_rate,        // Cache refresh rate
    const float gamma,               // Cache replacement ratio
    const int layer_id,              // Layer identifier
    int* refresh_indices,            // Output refresh indices
    int* new_neighbors,              // Output new neighbors
    curandState* states              // Random number states
);
```

**Key Features:**
- Partial cache refresh with configurable rate
- Dynamic cache management
- Layer-specific refresh strategies
- Efficient cache update operations

### 3. Graph Structure Masking (`graph_structure_mask_kernel`)

Implements dense graph filtering and memory reduction:

```cuda
__global__ void graph_structure_mask_kernel(
    const GraphData input_graph,     // Input graph
    GraphData output_graph,          // Output filtered graph
    const int* node_degree_thresholds, // Degree thresholds
    const int* edge_masks,           // Edge masks
    int* valid_nodes,                // Valid node count
    int* valid_edges,                // Valid edge count
    int* node_mapping,               // Node ID mapping
    int* edge_mapping                // Edge ID mapping
);
```

**Key Features:**
- Degree-based node filtering
- Edge masking for selective sampling
- Memory-efficient graph compression
- ID mapping for graph reconstruction

### 4. Multi-hop Aggregation (`multi_hop_aggregation_kernel`)

Implements efficient k-hop neighbor sampling:

```cuda
__global__ void multi_hop_aggregation_kernel(
    const int* seed_nodes,           // Input seed nodes
    const int num_seeds,             // Number of seed nodes
    const GraphData graph,           // Graph structure
    const int* hop_fanouts,          // Fanout per hop
    const int num_hops,              // Number of hops
    int* hop_neighbors,              // Output hop neighbors
    int* hop_counts,                 // Output hop counts
    int* visited_mask,               // Visited node mask
    curandState* states              // Random number states
);
```

**Key Features:**
- Multi-hop neighbor sampling
- Visited node tracking to avoid duplicates
- Configurable fanout per hop
- Efficient memory usage with bit masks

## Data Structures

### GraphData
```cuda
struct GraphData {
    int* src_nodes;      // Source nodes array
    int* dst_nodes;      // Destination nodes array
    float* edge_weights; // Edge weights for sampling
    int* node_degrees;   // Node degrees
    int* csr_indices;    // CSR format indices
    int* csr_offsets;    // CSR format offsets
    int num_nodes;       // Number of nodes
    int num_edges;       // Number of edges
};
```

### CacheData
```cuda
struct CacheData {
    int* cached_src;     // Cached source nodes
    int* cached_dst;     // Cached destination nodes
    float* cached_weights; // Cached edge weights
    int* cache_indices;  // Cache indices
    int cache_size;      // Current cache size
    int max_cache_size;  // Maximum cache size
};
```

## Performance Optimizations

### Memory Access Patterns
- **Coalesced Memory Access**: Threads access consecutive memory locations
- **Shared Memory Usage**: Frequently accessed data cached in shared memory
- **Memory Prefetching**: Data prefetched to reduce latency

### Compute Optimizations
- **Warp-Level Primitives**: Efficient intra-warp operations
- **Atomic Operations**: Thread-safe updates for shared data
- **Branch Divergence Minimization**: Reduced control flow divergence

### Cache Efficiency
- **Cache-Aware Data Layout**: Optimized memory layout for cache locality
- **LRU Cache Management**: Efficient cache eviction policies
- **Cache Amplification**: Pre-sampling for reduced I/O

## Build Instructions

### Prerequisites
- CUDA Toolkit 11.0 or later
- Python 3.7 or later
- PyTorch 1.9.0 or later
- pybind11 2.6.0 or later

### Building CUDA Kernels
```bash
cd cuda_kernels
python setup.py build_ext --inplace
```

### Building with Specific CUDA Version
```bash
export CUDA_HOME=/usr/local/cuda-11.0
python setup.py build_ext --inplace
```

## Usage Examples

### Python API Usage

```python
import torch
import graphsnapshot_cuda

# Create sample data
seed_nodes = torch.tensor([0, 1, 2, 3, 4], device='cuda')
src_nodes = torch.randint(0, 100, (1000,), device='cuda')
dst_nodes = torch.randint(0, 100, (1000,), device='cuda')
node_degrees = torch.randint(1, 50, (100,), device='cuda')

# FCR Neighbor Sampling
result = graphsnapshot_cuda.neighbor_sampling_fcr(
    seed_nodes=seed_nodes,
    src_nodes=src_nodes,
    dst_nodes=dst_nodes,
    edge_weights=torch.empty(0),  # No edge weights
    node_degrees=node_degrees,
    csr_indices=dst_nodes,
    csr_offsets=torch.cumsum(torch.cat([torch.tensor([0]), node_degrees]), 0),
    cached_src=torch.empty(0),
    cached_dst=torch.empty(0),
    cached_weights=torch.empty(0),
    cache_indices=torch.empty(0),
    fanouts=torch.tensor([10, 10, 10, 10, 10], device='cuda'),
    alpha=2.0,
    use_cache=False
)

sampled_neighbors = result[0]
neighbor_counts = result[1]
```

### C++ API Usage

```cpp
#include "graphsnapshot_kernels.h"

// Initialize graph data
GraphData graph;
graph.src_nodes = src_nodes_ptr;
graph.dst_nodes = dst_nodes_ptr;
graph.num_nodes = num_nodes;
graph.num_edges = num_edges;

// Launch kernel
launch_neighbor_sampling_fcr(
    seed_nodes_ptr, num_seeds, &graph, &cache,
    fanouts_ptr, alpha, use_cache,
    sampled_neighbors_ptr, neighbor_counts_ptr,
    states_ptr, stream
);
```

## Integration with GraphSnapShot

These CUDA kernels are designed to integrate seamlessly with the GraphSnapShot framework:

1. **FBL Integration**: Direct replacement for standard neighbor sampling
2. **OTF Integration**: Dynamic cache management for on-the-fly sampling
3. **FCR Integration**: Full cache refresh with amplification
4. **Memory Management**: GraphSnapShot memory optimization

## License

This code is part of the GraphSnapShot project and follows the same licensing terms.
