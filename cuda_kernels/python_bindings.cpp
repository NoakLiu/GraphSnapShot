/**
 * GraphSnapShot Python Bindings
 * 
 * Python bindings for GraphSnapShot CUDA kernels using pybind11
 * 
 * Author: GraphSnapShot Team
 * Date: 2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <vector>

#include "graphsnapshot_kernels.h"

namespace py = pybind11;

// Helper function to get device pointer from torch tensor
template<typename T>
T* get_tensor_ptr(torch::Tensor tensor) {
    return tensor.data_ptr<T>();
}

// Helper function to create GraphData from torch tensors
GraphData create_graph_data_from_tensors(
    torch::Tensor src_nodes,
    torch::Tensor dst_nodes,
    torch::Tensor edge_weights,
    torch::Tensor node_degrees,
    torch::Tensor csr_indices,
    torch::Tensor csr_offsets
) {
    GraphData graph;
    graph.src_nodes = get_tensor_ptr<int>(src_nodes);
    graph.dst_nodes = get_tensor_ptr<int>(dst_nodes);
    graph.edge_weights = edge_weights.numel() > 0 ? get_tensor_ptr<float>(edge_weights) : nullptr;
    graph.node_degrees = get_tensor_ptr<int>(node_degrees);
    graph.csr_indices = get_tensor_ptr<int>(csr_indices);
    graph.csr_offsets = get_tensor_ptr<int>(csr_offsets);
    graph.num_nodes = src_nodes.size(0);
    graph.num_edges = src_nodes.size(0);
    return graph;
}

// Helper function to create CacheData from torch tensors
CacheData create_cache_data_from_tensors(
    torch::Tensor cached_src,
    torch::Tensor cached_dst,
    torch::Tensor cached_weights,
    torch::Tensor cache_indices,
    int cache_size,
    int max_cache_size
) {
    CacheData cache;
    cache.cached_src = get_tensor_ptr<int>(cached_src);
    cache.cached_dst = get_tensor_ptr<int>(cached_dst);
    cache.cached_weights = cached_weights.numel() > 0 ? get_tensor_ptr<float>(cached_weights) : nullptr;
    cache.cache_indices = get_tensor_ptr<int>(cache_indices);
    cache.cache_size = cache_size;
    cache.max_cache_size = max_cache_size;
    return cache;
}

// Python wrapper for FCR neighbor sampling
torch::Tensor neighbor_sampling_fcr_py(
    torch::Tensor seed_nodes,
    torch::Tensor src_nodes,
    torch::Tensor dst_nodes,
    torch::Tensor edge_weights,
    torch::Tensor node_degrees,
    torch::Tensor csr_indices,
    torch::Tensor csr_offsets,
    torch::Tensor cached_src,
    torch::Tensor cached_dst,
    torch::Tensor cached_weights,
    torch::Tensor cache_indices,
    torch::Tensor fanouts,
    float alpha,
    bool use_cache,
    int max_fanout = 1024
) {
    // Ensure tensors are on GPU and contiguous
    auto device = seed_nodes.device();
    auto dtype = torch::kInt32;
    
    seed_nodes = seed_nodes.to(device).contiguous();
    src_nodes = src_nodes.to(device).contiguous();
    dst_nodes = dst_nodes.to(device).contiguous();
    node_degrees = node_degrees.to(device).contiguous();
    csr_indices = csr_indices.to(device).contiguous();
    csr_offsets = csr_offsets.to(device).contiguous();
    fanouts = fanouts.to(device).contiguous();
    
    if (edge_weights.numel() > 0) {
        edge_weights = edge_weights.to(device).contiguous();
    }
    if (use_cache) {
        cached_src = cached_src.to(device).contiguous();
        cached_dst = cached_dst.to(device).contiguous();
        if (cached_weights.numel() > 0) {
            cached_weights = cached_weights.to(device).contiguous();
        }
        cache_indices = cache_indices.to(device).contiguous();
    }
    
    int num_seeds = seed_nodes.size(0);
    
    // Create output tensors
    auto sampled_neighbors = torch::full({num_seeds, max_fanout}, -1, 
                                        torch::TensorOptions().dtype(dtype).device(device));
    auto neighbor_counts = torch::zeros({num_seeds}, 
                                       torch::TensorOptions().dtype(dtype).device(device));
    
    // Initialize random states
    auto states = torch::empty({num_seeds * max_fanout}, 
                              torch::TensorOptions().dtype(torch::kUInt32).device(device));
    initialize_curand_states(get_tensor_ptr<curandState>(states), 
                            num_seeds * max_fanout, 42);
    
    // Create data structures
    GraphData graph = create_graph_data_from_tensors(
        src_nodes, dst_nodes, edge_weights, node_degrees, csr_indices, csr_offsets
    );
    
    CacheData cache;
    if (use_cache) {
        cache = create_cache_data_from_tensors(
            cached_src, cached_dst, cached_weights, cache_indices, 
            cached_src.size(0), cached_src.size(0)
        );
    } else {
        cache.cached_src = nullptr;
        cache.cached_dst = nullptr;
        cache.cached_weights = nullptr;
        cache.cache_indices = nullptr;
        cache.cache_size = 0;
        cache.max_cache_size = 0;
    }
    
    // Launch kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    launch_neighbor_sampling_fcr(
        get_tensor_ptr<int>(seed_nodes),
        num_seeds,
        &graph,
        &cache,
        get_tensor_ptr<int>(fanouts),
        alpha,
        use_cache,
        get_tensor_ptr<int>(sampled_neighbors),
        get_tensor_ptr<int>(neighbor_counts),
        get_tensor_ptr<curandState>(states),
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return torch::stack({sampled_neighbors, neighbor_counts});
}

// Python wrapper for OTF cache refresh
torch::Tensor cache_refresh_otf_py(
    torch::Tensor seed_nodes,
    torch::Tensor src_nodes,
    torch::Tensor dst_nodes,
    torch::Tensor edge_weights,
    torch::Tensor node_degrees,
    torch::Tensor csr_indices,
    torch::Tensor csr_offsets,
    torch::Tensor cached_src,
    torch::Tensor cached_dst,
    torch::Tensor cached_weights,
    torch::Tensor cache_indices,
    float refresh_rate,
    float gamma,
    int layer_id,
    int max_fanout = 1024
) {
    auto device = seed_nodes.device();
    auto dtype = torch::kInt32;
    
    // Ensure tensors are on GPU and contiguous
    seed_nodes = seed_nodes.to(device).contiguous();
    src_nodes = src_nodes.to(device).contiguous();
    dst_nodes = dst_nodes.to(device).contiguous();
    node_degrees = node_degrees.to(device).contiguous();
    csr_indices = csr_indices.to(device).contiguous();
    csr_offsets = csr_offsets.to(device).contiguous();
    cached_src = cached_src.to(device).contiguous();
    cached_dst = cached_dst.to(device).contiguous();
    cache_indices = cache_indices.to(device).contiguous();
    
    if (edge_weights.numel() > 0) {
        edge_weights = edge_weights.to(device).contiguous();
    }
    if (cached_weights.numel() > 0) {
        cached_weights = cached_weights.to(device).contiguous();
    }
    
    int num_seeds = seed_nodes.size(0);
    
    // Create output tensors
    auto refresh_indices = torch::full({num_seeds, max_fanout}, -1, 
                                      torch::TensorOptions().dtype(dtype).device(device));
    auto new_neighbors = torch::full({num_seeds, max_fanout}, -1, 
                                    torch::TensorOptions().dtype(dtype).device(device));
    
    // Initialize random states
    auto states = torch::empty({num_seeds * max_fanout}, 
                              torch::TensorOptions().dtype(torch::kUInt32).device(device));
    initialize_curand_states(get_tensor_ptr<curandState>(states), 
                            num_seeds * max_fanout, 42);
    
    // Create data structures
    GraphData graph = create_graph_data_from_tensors(
        src_nodes, dst_nodes, edge_weights, node_degrees, csr_indices, csr_offsets
    );
    
    CacheData cache = create_cache_data_from_tensors(
        cached_src, cached_dst, cached_weights, cache_indices, 
        cached_src.size(0), cached_src.size(0)
    );
    
    // Launch kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    launch_cache_refresh_otf(
        get_tensor_ptr<int>(seed_nodes),
        num_seeds,
        &graph,
        &cache,
        refresh_rate,
        gamma,
        layer_id,
        get_tensor_ptr<int>(refresh_indices),
        get_tensor_ptr<int>(new_neighbors),
        get_tensor_ptr<curandState>(states),
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return torch::stack({refresh_indices, new_neighbors});
}

// Python wrapper for graph structure masking
std::vector<torch::Tensor> graph_structure_mask_py(
    torch::Tensor src_nodes,
    torch::Tensor dst_nodes,
    torch::Tensor edge_weights,
    torch::Tensor node_degrees,
    torch::Tensor csr_indices,
    torch::Tensor csr_offsets,
    torch::Tensor node_degree_thresholds,
    torch::Tensor edge_masks
) {
    auto device = src_nodes.device();
    auto dtype = torch::kInt32;
    
    // Ensure tensors are on GPU and contiguous
    src_nodes = src_nodes.to(device).contiguous();
    dst_nodes = dst_nodes.to(device).contiguous();
    node_degrees = node_degrees.to(device).contiguous();
    csr_indices = csr_indices.to(device).contiguous();
    csr_offsets = csr_offsets.to(device).contiguous();
    node_degree_thresholds = node_degree_thresholds.to(device).contiguous();
    edge_masks = edge_masks.to(device).contiguous();
    
    if (edge_weights.numel() > 0) {
        edge_weights = edge_weights.to(device).contiguous();
    }
    
    int num_nodes = src_nodes.size(0);
    int num_edges = src_nodes.size(0);
    
    // Create output tensors
    auto output_src = torch::empty({num_edges}, torch::TensorOptions().dtype(dtype).device(device));
    auto output_dst = torch::empty({num_edges}, torch::TensorOptions().dtype(dtype).device(device));
    auto output_weights = torch::empty({num_edges}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto valid_nodes = torch::zeros({1}, torch::TensorOptions().dtype(dtype).device(device));
    auto valid_edges = torch::zeros({1}, torch::TensorOptions().dtype(dtype).device(device));
    auto node_mapping = torch::full({num_nodes}, -1, torch::TensorOptions().dtype(dtype).device(device));
    auto edge_mapping = torch::full({num_edges}, -1, torch::TensorOptions().dtype(dtype).device(device));
    
    // Create data structures
    GraphData input_graph = create_graph_data_from_tensors(
        src_nodes, dst_nodes, edge_weights, node_degrees, csr_indices, csr_offsets
    );
    
    GraphData output_graph;
    output_graph.src_nodes = get_tensor_ptr<int>(output_src);
    output_graph.dst_nodes = get_tensor_ptr<int>(output_dst);
    output_graph.edge_weights = get_tensor_ptr<float>(output_weights);
    output_graph.node_degrees = nullptr;
    output_graph.csr_indices = nullptr;
    output_graph.csr_offsets = nullptr;
    output_graph.num_nodes = num_nodes;
    output_graph.num_edges = num_edges;
    
    // Launch kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    launch_graph_structure_mask(
        &input_graph,
        &output_graph,
        get_tensor_ptr<int>(node_degree_thresholds),
        get_tensor_ptr<int>(edge_masks),
        get_tensor_ptr<int>(valid_nodes),
        get_tensor_ptr<int>(valid_edges),
        get_tensor_ptr<int>(node_mapping),
        get_tensor_ptr<int>(edge_mapping),
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return {output_src, output_dst, output_weights, valid_nodes, valid_edges, node_mapping, edge_mapping};
}

// Python wrapper for multi-hop aggregation
torch::Tensor multi_hop_aggregation_py(
    torch::Tensor seed_nodes,
    torch::Tensor src_nodes,
    torch::Tensor dst_nodes,
    torch::Tensor node_degrees,
    torch::Tensor csr_indices,
    torch::Tensor csr_offsets,
    torch::Tensor hop_fanouts,
    int num_hops,
    int max_fanout = 1024
) {
    auto device = seed_nodes.device();
    auto dtype = torch::kInt32;
    
    // Ensure tensors are on GPU and contiguous
    seed_nodes = seed_nodes.to(device).contiguous();
    src_nodes = src_nodes.to(device).contiguous();
    dst_nodes = dst_nodes.to(device).contiguous();
    node_degrees = node_degrees.to(device).contiguous();
    csr_indices = csr_indices.to(device).contiguous();
    csr_offsets = csr_offsets.to(device).contiguous();
    hop_fanouts = hop_fanouts.to(device).contiguous();
    
    int num_seeds = seed_nodes.size(0);
    int num_nodes = src_nodes.size(0);
    
    // Create output tensors
    auto hop_neighbors = torch::full({num_seeds, num_hops, max_fanout}, -1, 
                                    torch::TensorOptions().dtype(dtype).device(device));
    auto hop_counts = torch::zeros({num_seeds, num_hops}, 
                                  torch::TensorOptions().dtype(dtype).device(device));
    auto visited_mask = torch::zeros({num_seeds, num_nodes}, 
                                    torch::TensorOptions().dtype(dtype).device(device));
    
    // Initialize random states
    auto states = torch::empty({num_seeds * max_fanout}, 
                              torch::TensorOptions().dtype(torch::kUInt32).device(device));
    initialize_curand_states(get_tensor_ptr<curandState>(states), 
                            num_seeds * max_fanout, 42);
    
    // Create data structures
    GraphData graph;
    graph.src_nodes = get_tensor_ptr<int>(src_nodes);
    graph.dst_nodes = get_tensor_ptr<int>(dst_nodes);
    graph.edge_weights = nullptr;
    graph.node_degrees = get_tensor_ptr<int>(node_degrees);
    graph.csr_indices = get_tensor_ptr<int>(csr_indices);
    graph.csr_offsets = get_tensor_ptr<int>(csr_offsets);
    graph.num_nodes = num_nodes;
    graph.num_edges = src_nodes.size(0);
    
    // Launch kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    launch_multi_hop_aggregation(
        get_tensor_ptr<int>(seed_nodes),
        num_seeds,
        &graph,
        get_tensor_ptr<int>(hop_fanouts),
        num_hops,
        get_tensor_ptr<int>(hop_neighbors),
        get_tensor_ptr<int>(hop_counts),
        get_tensor_ptr<int>(visited_mask),
        get_tensor_ptr<curandState>(states),
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return torch::stack({hop_neighbors, hop_counts, visited_mask});
}

// Python wrapper for heterogeneous sampling
torch::Tensor heterogeneous_sampling_py(
    torch::Tensor seed_nodes,
    torch::Tensor src_nodes,
    torch::Tensor dst_nodes,
    torch::Tensor node_degrees,
    torch::Tensor csr_indices,
    torch::Tensor csr_offsets,
    torch::Tensor node_types,
    torch::Tensor edge_types,
    torch::Tensor fanouts_per_type,
    int num_edge_types,
    int max_fanout = 1024
) {
    auto device = seed_nodes.device();
    auto dtype = torch::kInt32;
    
    // Ensure tensors are on GPU and contiguous
    seed_nodes = seed_nodes.to(device).contiguous();
    src_nodes = src_nodes.to(device).contiguous();
    dst_nodes = dst_nodes.to(device).contiguous();
    node_degrees = node_degrees.to(device).contiguous();
    csr_indices = csr_indices.to(device).contiguous();
    csr_offsets = csr_offsets.to(device).contiguous();
    node_types = node_types.to(device).contiguous();
    edge_types = edge_types.to(device).contiguous();
    fanouts_per_type = fanouts_per_type.to(device).contiguous();
    
    int num_seeds = seed_nodes.size(0);
    
    // Create output tensors
    auto sampled_neighbors = torch::full({num_seeds, max_fanout}, -1, 
                                        torch::TensorOptions().dtype(dtype).device(device));
    auto neighbor_counts = torch::zeros({num_seeds}, 
                                       torch::TensorOptions().dtype(dtype).device(device));
    auto type_counts = torch::zeros({num_seeds, num_edge_types}, 
                                   torch::TensorOptions().dtype(dtype).device(device));
    
    // Initialize random states
    auto states = torch::empty({num_seeds * max_fanout}, 
                              torch::TensorOptions().dtype(torch::kUInt32).device(device));
    initialize_curand_states(get_tensor_ptr<curandState>(states), 
                            num_seeds * max_fanout, 42);
    
    // Create data structures
    GraphData graph;
    graph.src_nodes = get_tensor_ptr<int>(src_nodes);
    graph.dst_nodes = get_tensor_ptr<int>(dst_nodes);
    graph.edge_weights = nullptr;
    graph.node_degrees = get_tensor_ptr<int>(node_degrees);
    graph.csr_indices = get_tensor_ptr<int>(csr_indices);
    graph.csr_offsets = get_tensor_ptr<int>(csr_offsets);
    graph.num_nodes = src_nodes.size(0);
    graph.num_edges = src_nodes.size(0);
    
    // Launch kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    launch_heterogeneous_sampling(
        get_tensor_ptr<int>(seed_nodes),
        num_seeds,
        &graph,
        get_tensor_ptr<int>(node_types),
        get_tensor_ptr<int>(edge_types),
        get_tensor_ptr<int>(fanouts_per_type),
        num_edge_types,
        get_tensor_ptr<int>(sampled_neighbors),
        get_tensor_ptr<int>(neighbor_counts),
        get_tensor_ptr<int>(type_counts),
        get_tensor_ptr<curandState>(states),
        stream
    );
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return torch::stack({sampled_neighbors, neighbor_counts, type_counts});
}

PYBIND11_MODULE(graphsnapshot_cuda, m) {
    m.doc() = "GraphSnapShot CUDA Kernels";
    
    // Core sampling functions
    m.def("neighbor_sampling_fcr", &neighbor_sampling_fcr_py,
          "Full Cache Refresh neighbor sampling",
          py::arg("seed_nodes"),
          py::arg("src_nodes"),
          py::arg("dst_nodes"),
          py::arg("edge_weights"),
          py::arg("node_degrees"),
          py::arg("csr_indices"),
          py::arg("csr_offsets"),
          py::arg("cached_src"),
          py::arg("cached_dst"),
          py::arg("cached_weights"),
          py::arg("cache_indices"),
          py::arg("fanouts"),
          py::arg("alpha"),
          py::arg("use_cache"),
          py::arg("max_fanout") = 1024);
    
    m.def("cache_refresh_otf", &cache_refresh_otf_py,
          "On-The-Fly cache refresh",
          py::arg("seed_nodes"),
          py::arg("src_nodes"),
          py::arg("dst_nodes"),
          py::arg("edge_weights"),
          py::arg("node_degrees"),
          py::arg("csr_indices"),
          py::arg("csr_offsets"),
          py::arg("cached_src"),
          py::arg("cached_dst"),
          py::arg("cached_weights"),
          py::arg("cache_indices"),
          py::arg("refresh_rate"),
          py::arg("gamma"),
          py::arg("layer_id"),
          py::arg("max_fanout") = 1024);
    
    m.def("graph_structure_mask", &graph_structure_mask_py,
          "Graph structure masking and filtering",
          py::arg("src_nodes"),
          py::arg("dst_nodes"),
          py::arg("edge_weights"),
          py::arg("node_degrees"),
          py::arg("csr_indices"),
          py::arg("csr_offsets"),
          py::arg("node_degree_thresholds"),
          py::arg("edge_masks"));
    
    m.def("multi_hop_aggregation", &multi_hop_aggregation_py,
          "Multi-hop neighbor aggregation",
          py::arg("seed_nodes"),
          py::arg("src_nodes"),
          py::arg("dst_nodes"),
          py::arg("node_degrees"),
          py::arg("csr_indices"),
          py::arg("csr_offsets"),
          py::arg("hop_fanouts"),
          py::arg("num_hops"),
          py::arg("max_fanout") = 1024);
    
    m.def("heterogeneous_sampling", &heterogeneous_sampling_py,
          "Heterogeneous graph sampling",
          py::arg("seed_nodes"),
          py::arg("src_nodes"),
          py::arg("dst_nodes"),
          py::arg("node_degrees"),
          py::arg("csr_indices"),
          py::arg("csr_offsets"),
          py::arg("node_types"),
          py::arg("edge_types"),
          py::arg("fanouts_per_type"),
          py::arg("num_edge_types"),
          py::arg("max_fanout") = 1024);
}
