/**
 * GraphSnapShot CUDA Kernels
 * 
 * Core CUDA kernels for graph sampling, caching, and memory management
 * supporting FBL, OTF, and FCR sampling strategies.
 * 
 * Author: GraphSnapShot Team
 * Date: 2025
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// Constants
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_FANOUT 1024
#define CACHE_SIZE 65536

// Data structures for graph representation
struct GraphData {
    int* src_nodes;      // Source nodes array
    int* dst_nodes;      // Destination nodes array
    float* edge_weights; // Edge weights for sampling
    int* node_degrees;   // Node degrees
    int* csr_indices;    // CSR format indices
    int* csr_offsets;    // CSR format offsets
    int num_nodes;
    int num_edges;
};

struct CacheData {
    int* cached_src;     // Cached source nodes
    int* cached_dst;     // Cached destination nodes
    float* cached_weights; // Cached edge weights
    int* cache_indices;  // Cache indices
    int cache_size;
    int max_cache_size;
};

// Utility functions
__device__ __forceinline__ int get_thread_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int get_warp_id() {
    return threadIdx.x / WARP_SIZE;
}

__device__ __forceinline__ int get_lane_id() {
    return threadIdx.x % WARP_SIZE;
}

/**
 * Kernel 1: Neighbor Sampling with Cache Lookup (FCR Strategy)
 * 
 * This kernel implements the core neighbor sampling with cache amplification
 * for the Full Cache Refresh (FCR) strategy.
 */
__global__ void neighbor_sampling_fcr_kernel(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData graph,
    const CacheData cache,
    const int* fanouts,
    const float alpha,
    const bool use_cache,
    int* sampled_neighbors,
    int* neighbor_counts,
    curandState* states
) {
    int tid = get_thread_idx();
    int seed_id = tid / MAX_FANOUT;
    int sample_id = tid % MAX_FANOUT;
    
    if (seed_id >= num_seeds) return;
    
    int seed_node = seed_nodes[seed_id];
    int fanout = fanouts[seed_id];
    int amplified_fanout = (int)(fanout * alpha);
    
    // Initialize neighbor count for this seed
    if (sample_id == 0) {
        neighbor_counts[seed_id] = 0;
    }
    __syncthreads();
    
    // Sample from cache if available and enabled
    if (use_cache && cache.cache_size > 0) {
        int cache_start = seed_node * cache.max_cache_size;
        int cached_degree = min(cache.max_cache_size, graph.node_degrees[seed_node]);
        
        if (sample_id < cached_degree && sample_id < amplified_fanout) {
            int neighbor = cache.cached_dst[cache_start + sample_id];
            sampled_neighbors[seed_id * MAX_FANOUT + sample_id] = neighbor;
            
            // Atomic increment of neighbor count
            atomicAdd(&neighbor_counts[seed_id], 1);
        }
    }
    
    // Sample additional neighbors from original graph if needed
    int remaining_fanout = amplified_fanout - neighbor_counts[seed_id];
    if (sample_id < remaining_fanout && sample_id < graph.node_degrees[seed_node]) {
        int csr_start = graph.csr_offsets[seed_node];
        int csr_end = graph.csr_offsets[seed_node + 1];
        
        if (csr_start + sample_id < csr_end) {
            // Use curand for random sampling
            curandState local_state = states[tid];
            float rand_val = curand_uniform(&local_state);
            int neighbor_idx = csr_start + (int)(rand_val * (csr_end - csr_start));
            int neighbor = graph.csr_indices[neighbor_idx];
            
            sampled_neighbors[seed_id * MAX_FANOUT + neighbor_counts[seed_id] + sample_id] = neighbor;
            atomicAdd(&neighbor_counts[seed_id], 1);
            states[tid] = local_state;
        }
    }
}

/**
 * Kernel 2: Cache Refresh (OTF Strategy)
 * 
 * This kernel implements partial cache refresh for On-The-Fly (OTF) sampling,
 * replacing a portion of cached edges with fresh samples.
 */
__global__ void cache_refresh_otf_kernel(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData graph,
    CacheData cache,
    const float refresh_rate,
    const float gamma,
    const int layer_id,
    int* refresh_indices,
    int* new_neighbors,
    curandState* states
) {
    int tid = get_thread_idx();
    int seed_id = tid / MAX_FANOUT;
    int sample_id = tid % MAX_FANOUT;
    
    if (seed_id >= num_seeds) return;
    
    int seed_node = seed_nodes[seed_id];
    int cache_start = seed_node * cache.max_cache_size;
    int cached_degree = min(cache.max_cache_size, graph.node_degrees[seed_node]);
    
    // Calculate how many edges to refresh
    int edges_to_refresh = (int)(cached_degree * gamma * refresh_rate);
    
    if (sample_id < edges_to_refresh) {
        // Mark edge for removal from cache
        int edge_to_remove = cache_start + sample_id;
        refresh_indices[seed_id * MAX_FANOUT + sample_id] = edge_to_remove;
        
        // Sample new neighbor to replace the removed one
        curandState local_state = states[tid];
        float rand_val = curand_uniform(&local_state);
        
        int csr_start = graph.csr_offsets[seed_node];
        int csr_end = graph.csr_offsets[seed_node + 1];
        int neighbor_idx = csr_start + (int)(rand_val * (csr_end - csr_start));
        int new_neighbor = graph.csr_indices[neighbor_idx];
        
        new_neighbors[seed_id * MAX_FANOUT + sample_id] = new_neighbor;
        states[tid] = local_state;
    }
}

/**
 * Kernel 3: Graph Structure Masking and Filtering
 * 
 * This kernel implements dense graph filtering and structure masking
 * for memory reduction in GraphSnapShot.
 */
__global__ void graph_structure_mask_kernel(
    const GraphData input_graph,
    GraphData output_graph,
    const int* node_degree_thresholds,
    const int* edge_masks,
    int* valid_nodes,
    int* valid_edges,
    int* node_mapping,
    int* edge_mapping
) {
    int tid = get_thread_idx();
    
    // Filter nodes based on degree threshold
    if (tid < input_graph.num_nodes) {
        int degree = input_graph.node_degrees[tid];
        int threshold = node_degree_thresholds[tid];
        
        if (degree > threshold) {
            int valid_idx = atomicAdd(valid_nodes, 1);
            node_mapping[tid] = valid_idx;
        } else {
            node_mapping[tid] = -1; // Invalid node
        }
    }
    
    __syncthreads();
    
    // Filter edges based on masks and valid nodes
    if (tid < input_graph.num_edges) {
        int src = input_graph.src_nodes[tid];
        int dst = input_graph.dst_nodes[tid];
        int mask = edge_masks[tid];
        
        // Check if both nodes are valid and edge is not masked
        if (mask == 1 && node_mapping[src] != -1 && node_mapping[dst] != -1) {
            int valid_edge_idx = atomicAdd(valid_edges, 1);
            edge_mapping[tid] = valid_edge_idx;
            
            // Update output graph structure
            output_graph.src_nodes[valid_edge_idx] = node_mapping[src];
            output_graph.dst_nodes[valid_edge_idx] = node_mapping[dst];
            if (output_graph.edge_weights) {
                output_graph.edge_weights[valid_edge_idx] = input_graph.edge_weights[tid];
            }
        } else {
            edge_mapping[tid] = -1; // Invalid edge
        }
    }
}

/**
 * Kernel 4: Buffer Management and LRU Cache Update
 * 
 * This kernel manages the buffer cache with LRU eviction policy
 * for efficient memory usage in GraphSnapShot.
 */
__global__ void buffer_cache_update_kernel(
    CacheData cache,
    const int* access_keys,
    const int num_accesses,
    int* cache_indices,
    int* access_times,
    int* lru_counters,
    int current_time,
    const int cache_capacity
) {
    int tid = get_thread_idx();
    
    if (tid >= num_accesses) return;
    
    int key = access_keys[tid];
    int cache_idx = cache_indices[key];
    
    if (cache_idx != -1) {
        // Cache hit - update access time
        access_times[cache_idx] = current_time;
        lru_counters[cache_idx] = atomicAdd(lru_counters, 1);
    } else {
        // Cache miss - need to find slot or evict
        if (cache.cache_size < cache_capacity) {
            // Cache not full - add new entry
            cache_idx = atomicAdd(&cache.cache_size, 1);
            cache_indices[key] = cache_idx;
            access_times[cache_idx] = current_time;
            lru_counters[cache_idx] = atomicAdd(lru_counters, 1);
        } else {
            // Cache full - find LRU entry to evict
            // This is a simplified version; in practice, you'd use a more
            // sophisticated data structure for O(1) LRU operations
            int min_time = INT_MAX;
            int lru_idx = -1;
            
            for (int i = 0; i < cache_capacity; i++) {
                if (access_times[i] < min_time) {
                    min_time = access_times[i];
                    lru_idx = i;
                }
            }
            
            if (lru_idx != -1) {
                // Evict LRU entry and replace with new one
                cache_indices[key] = lru_idx;
                access_times[lru_idx] = current_time;
                lru_counters[lru_idx] = atomicAdd(lru_counters, 1);
            }
        }
    }
}

/**
 * Kernel 5: Multi-hop Neighbor Aggregation
 * 
 * This kernel performs efficient multi-hop neighbor aggregation
 * for k-hop sampling in GraphSnapShot.
 */
__global__ void multi_hop_aggregation_kernel(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData graph,
    const int* hop_fanouts,
    const int num_hops,
    int* hop_neighbors,
    int* hop_counts,
    int* visited_mask,
    curandState* states
) {
    int tid = get_thread_idx();
    int seed_id = tid / MAX_FANOUT;
    int sample_id = tid % MAX_FANOUT;
    
    if (seed_id >= num_seeds) return;
    
    int seed_node = seed_nodes[seed_id];
    
    // Initialize visited mask
    if (sample_id == 0) {
        visited_mask[seed_id * graph.num_nodes + seed_node] = 1;
    }
    
    for (int hop = 0; hop < num_hops; hop++) {
        int fanout = hop_fanouts[hop];
        int hop_start = seed_id * MAX_FANOUT * num_hops + hop * MAX_FANOUT;
        
        if (sample_id < fanout) {
            // Get current layer nodes
            int current_node;
            if (hop == 0) {
                current_node = seed_node;
            } else {
                current_node = hop_neighbors[hop_start - MAX_FANOUT + sample_id];
            }
            
            if (current_node != -1) {
                // Sample neighbors for current hop
                curandState local_state = states[tid];
                float rand_val = curand_uniform(&local_state);
                
                int csr_start = graph.csr_offsets[current_node];
                int csr_end = graph.csr_offsets[current_node + 1];
                int neighbor_idx = csr_start + (int)(rand_val * (csr_end - csr_start));
                int neighbor = graph.csr_indices[neighbor_idx];
                
                // Check if neighbor is already visited
                int is_visited = visited_mask[seed_id * graph.num_nodes + neighbor];
                if (!is_visited) {
                    hop_neighbors[hop_start + sample_id] = neighbor;
                    visited_mask[seed_id * graph.num_nodes + neighbor] = 1;
                    atomicAdd(&hop_counts[seed_id * num_hops + hop], 1);
                }
                
                states[tid] = local_state;
            }
        }
        
        __syncthreads();
    }
}

/**
 * Kernel 6: Edge Weight Sampling with Probabilities
 * 
 * This kernel implements weighted edge sampling for heterogeneous graphs
 * and probability-based neighbor selection.
 */
__global__ void weighted_edge_sampling_kernel(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData graph,
    const float* edge_probabilities,
    const int fanout,
    int* sampled_neighbors,
    int* neighbor_counts,
    curandState* states
) {
    int tid = get_thread_idx();
    int seed_id = tid / MAX_FANOUT;
    int sample_id = tid % MAX_FANOUT;
    
    if (seed_id >= num_seeds || sample_id >= fanout) return;
    
    int seed_node = seed_nodes[seed_id];
    int csr_start = graph.csr_offsets[seed_node];
    int csr_end = graph.csr_offsets[seed_node + 1];
    int degree = csr_end - csr_start;
    
    if (degree == 0) return;
    
    curandState local_state = states[tid];
    
    // Use alias method for efficient weighted sampling
    // This is a simplified version; full alias method would be more complex
    float total_weight = 0.0f;
    for (int i = csr_start; i < csr_end; i++) {
        total_weight += edge_probabilities[i];
    }
    
    float rand_val = curand_uniform(&local_state) * total_weight;
    float cumsum = 0.0f;
    
    for (int i = csr_start; i < csr_end; i++) {
        cumsum += edge_probabilities[i];
        if (cumsum >= rand_val) {
            int neighbor = graph.csr_indices[i];
            sampled_neighbors[seed_id * MAX_FANOUT + sample_id] = neighbor;
            atomicAdd(&neighbor_counts[seed_id], 1);
            break;
        }
    }
    
    states[tid] = local_state;
}

/**
 * Kernel 7: Memory Usage Detection and Optimization
 * 
 * This kernel monitors memory usage and performs memory optimization
 * for GraphSnapShot's dynamic memory management.
 */
__global__ void memory_usage_detection_kernel(
    const GraphData graph,
    const CacheData cache,
    int* memory_usage_stats,
    int* optimization_flags,
    const int memory_threshold,
    const int cache_threshold
) {
    int tid = get_thread_idx();
    
    if (tid == 0) {
        // Calculate memory usage statistics
        int graph_memory = graph.num_nodes * sizeof(int) * 2 + // src, dst arrays
                          graph.num_edges * sizeof(float) +    // edge weights
                          graph.num_nodes * sizeof(int) * 2;   // degrees, offsets
        
        int cache_memory = cache.cache_size * sizeof(int) * 2 + // cached src, dst
                          cache.cache_size * sizeof(float) +    // cached weights
                          cache.max_cache_size * sizeof(int);   // cache indices
        
        int total_memory = graph_memory + cache_memory;
        
        memory_usage_stats[0] = graph_memory;
        memory_usage_stats[1] = cache_memory;
        memory_usage_stats[2] = total_memory;
        
        // Set optimization flags
        optimization_flags[0] = (total_memory > memory_threshold) ? 1 : 0;
        optimization_flags[1] = (cache.cache_size > cache_threshold) ? 1 : 0;
        optimization_flags[2] = (graph.num_nodes > 100000) ? 1 : 0; // Large graph flag
    }
}

/**
 * Kernel 8: Heterogeneous Graph Sampling
 * 
 * This kernel handles sampling for heterogeneous graphs with different
 * node and edge types in GraphSnapShot.
 */
__global__ void heterogeneous_sampling_kernel(
    const int* seed_nodes,
    const int num_seeds,
    const GraphData graph,
    const int* node_types,
    const int* edge_types,
    const int* fanouts_per_type,
    const int num_edge_types,
    int* sampled_neighbors,
    int* neighbor_counts,
    int* type_counts,
    curandState* states
) {
    int tid = get_thread_idx();
    int seed_id = tid / MAX_FANOUT;
    int sample_id = tid % MAX_FANOUT;
    
    if (seed_id >= num_seeds) return;
    
    int seed_node = seed_nodes[seed_id];
    int seed_type = node_types[seed_node];
    int csr_start = graph.csr_offsets[seed_node];
    int csr_end = graph.csr_offsets[seed_node + 1];
    
    curandState local_state = states[tid];
    
    // Sample neighbors for each edge type
    for (int edge_type = 0; edge_type < num_edge_types; edge_type++) {
        int fanout_for_type = fanouts_per_type[edge_type];
        
        if (sample_id < fanout_for_type) {
            // Find edges of this type
            int type_start = -1, type_end = -1;
            for (int i = csr_start; i < csr_end; i++) {
                if (edge_types[i] == edge_type) {
                    if (type_start == -1) type_start = i;
                    type_end = i;
                }
            }
            
            if (type_start != -1) {
                int type_degree = type_end - type_start + 1;
                float rand_val = curand_uniform(&local_state);
                int neighbor_idx = type_start + (int)(rand_val * type_degree);
                int neighbor = graph.csr_indices[neighbor_idx];
                
                int global_sample_idx = seed_id * MAX_FANOUT + 
                                      atomicAdd(&neighbor_counts[seed_id], 1);
                sampled_neighbors[global_sample_idx] = neighbor;
                atomicAdd(&type_counts[seed_id * num_edge_types + edge_type], 1);
            }
        }
    }
    
    states[tid] = local_state;
}

// Host-side wrapper functions for kernel launches
extern "C" {

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
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_seeds * MAX_FANOUT + block.x - 1) / block.x);
    
    neighbor_sampling_fcr_kernel<<<grid, block, 0, stream>>>(
        seed_nodes, num_seeds, *graph, *cache, fanouts, alpha, use_cache,
        sampled_neighbors, neighbor_counts, states
    );
}

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
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_seeds * MAX_FANOUT + block.x - 1) / block.x);
    
    cache_refresh_otf_kernel<<<grid, block, 0, stream>>>(
        seed_nodes, num_seeds, *graph, *cache, refresh_rate, gamma, layer_id,
        refresh_indices, new_neighbors, states
    );
}

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
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((input_graph->num_edges + block.x - 1) / block.x);
    
    graph_structure_mask_kernel<<<grid, block, 0, stream>>>(
        *input_graph, *output_graph, node_degree_thresholds, edge_masks,
        valid_nodes, valid_edges, node_mapping, edge_mapping
    );
}

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
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_seeds * MAX_FANOUT + block.x - 1) / block.x);
    
    multi_hop_aggregation_kernel<<<grid, block, 0, stream>>>(
        seed_nodes, num_seeds, *graph, hop_fanouts, num_hops,
        hop_neighbors, hop_counts, visited_mask, states
    );
}

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
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_seeds * MAX_FANOUT + block.x - 1) / block.x);
    
    heterogeneous_sampling_kernel<<<grid, block, 0, stream>>>(
        seed_nodes, num_seeds, *graph, node_types, edge_types,
        fanouts_per_type, num_edge_types, sampled_neighbors,
        neighbor_counts, type_counts, states
    );
}

} // extern "C"
