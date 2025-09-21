/**
 * GraphSnapShot CUDA Kernels Header
 * 
 * Header file for GraphSnapShot CUDA kernel functions
 * 
 * Author: GraphSnapShot Team
 * Date: 2025
 */

#ifndef GRAPHSNAPSHOT_KERNELS_H
#define GRAPHSNAPSHOT_KERNELS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

// Data structures for graph representation
typedef struct {
    int* src_nodes;      // Source nodes array
    int* dst_nodes;      // Destination nodes array
    float* edge_weights; // Edge weights for sampling
    int* node_degrees;   // Node degrees
    int* csr_indices;    // CSR format indices
    int* csr_offsets;    // CSR format offsets
    int num_nodes;
    int num_edges;
} GraphData;

typedef struct {
    int* cached_src;     // Cached source nodes
    int* cached_dst;     // Cached destination nodes
    float* cached_weights; // Cached edge weights
    int* cache_indices;  // Cache indices
    int cache_size;
    int max_cache_size;
} CacheData;

// Kernel launch functions
void launch_neighbor_sampling_fcr(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData* graph,
    const CacheData* cache,
    const int* fanouts,
    const float alpha,
    const bool use_cache,
    int* sampled_neighbors,
    int* neighbor_counts,
    curandState* states,
    cudaStream_t stream
);

void launch_cache_refresh_otf(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData* graph,
    CacheData* cache,
    const float refresh_rate,
    const float gamma,
    const int layer_id,
    int* refresh_indices,
    int* new_neighbors,
    curandState* states,
    cudaStream_t stream
);

void launch_graph_structure_mask(
    const GraphData* input_graph,
    GraphData* output_graph,
    const int* node_degree_thresholds,
    const int* edge_masks,
    int* valid_nodes,
    int* valid_edges,
    int* node_mapping,
    int* edge_mapping,
    cudaStream_t stream
);

void launch_buffer_cache_update(
    CacheData* cache,
    const int* access_keys,
    const int num_accesses,
    int* cache_indices,
    int* access_times,
    int* lru_counters,
    int current_time,
    const int cache_capacity,
    cudaStream_t stream
);

void launch_multi_hop_aggregation(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData* graph,
    const int* hop_fanouts,
    const int num_hops,
    int* hop_neighbors,
    int* hop_counts,
    int* visited_mask,
    curandState* states,
    cudaStream_t stream
);

void launch_weighted_edge_sampling(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData* graph,
    const float* edge_probabilities,
    const int fanout,
    int* sampled_neighbors,
    int* neighbor_counts,
    curandState* states,
    cudaStream_t stream
);

void launch_memory_usage_detection(
    const GraphData* graph,
    const CacheData* cache,
    int* memory_usage_stats,
    int* optimization_flags,
    const int memory_threshold,
    const int cache_threshold,
    cudaStream_t stream
);

void launch_heterogeneous_sampling(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData* graph,
    const int* node_types,
    const int* edge_types,
    const int* fanouts_per_type,
    const int num_edge_types,
    int* sampled_neighbors,
    int* neighbor_counts,
    int* type_counts,
    curandState* states,
    cudaStream_t stream
);

// Utility functions
cudaError_t initialize_curand_states(curandState* states, int num_states, unsigned long seed);
cudaError_t cleanup_graph_data(GraphData* graph);
cudaError_t cleanup_cache_data(CacheData* cache);

#ifdef __cplusplus
}
#endif

#endif // GRAPHSNAPSHOT_KERNELS_H
