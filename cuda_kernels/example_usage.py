#!/usr/bin/env python3
"""
GraphSnapShot CUDA Kernels Example Usage

This script demonstrates how to use the GraphSnapShot CUDA kernels
for various graph sampling and caching operations.

Author: GraphSnapShot Team
Date: 2025
"""

import torch
import numpy as np
import time
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import graphsnapshot_cuda
    CUDA_AVAILABLE = True
    print("✓ GraphSnapShot CUDA kernels loaded successfully")
except ImportError:
    try:
        import graphsnapshot_cpu
        CUDA_AVAILABLE = False
        print("⚠ CUDA kernels not available, using CPU fallback")
    except ImportError:
        print("❌ No GraphSnapShot kernels available")
        sys.exit(1)

def create_sample_graph(num_nodes=1000, num_edges=5000, device='cuda'):
    """Create a sample graph for testing"""
    print(f"Creating sample graph with {num_nodes} nodes and {num_edges} edges...")
    
    # Generate random edges
    src_nodes = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst_nodes = torch.randint(0, num_nodes, (num_edges,), device=device)
    
    # Remove self-loops
    mask = src_nodes != dst_nodes
    src_nodes = src_nodes[mask]
    dst_nodes = dst_nodes[mask]
    
    # Calculate node degrees
    node_degrees = torch.zeros(num_nodes, device=device)
    for src in src_nodes:
        node_degrees[src] += 1
    
    # Create CSR representation
    csr_indices = dst_nodes
    csr_offsets = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    csr_offsets[1:] = torch.cumsum(node_degrees, dim=0)
    
    return src_nodes, dst_nodes, node_degrees, csr_indices, csr_offsets

def example_fcr_sampling():
    """Example of Full Cache Refresh (FCR) neighbor sampling"""
    print("\n" + "="*60)
    print("Example 1: Full Cache Refresh (FCR) Neighbor Sampling")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create sample graph
    src_nodes, dst_nodes, node_degrees, csr_indices, csr_offsets = create_sample_graph(
        num_nodes=1000, num_edges=5000, device=device
    )
    
    # Create seed nodes
    num_seeds = 100
    seed_nodes = torch.randint(0, 1000, (num_seeds,), device=device)
    fanouts = torch.full((num_seeds,), 10, device=device)
    
    # Create empty cache (no cache used in this example)
    empty_tensor = torch.empty(0, device=device)
    
    # Run FCR sampling
    print("Running FCR neighbor sampling...")
    start_time = time.time()
    
    if CUDA_AVAILABLE:
        result = graphsnapshot_cuda.neighbor_sampling_fcr(
            seed_nodes=seed_nodes,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            edge_weights=empty_tensor,
            node_degrees=node_degrees,
            csr_indices=csr_indices,
            csr_offsets=csr_offsets,
            cached_src=empty_tensor,
            cached_dst=empty_tensor,
            cached_weights=empty_tensor,
            cache_indices=empty_tensor,
            fanouts=fanouts,
            alpha=2.0,  # Cache amplification factor
            use_cache=False,
            max_fanout=1024
        )
    else:
        result = graphsnapshot_cpu.neighbor_sampling_fcr(
            seed_nodes=seed_nodes,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            edge_weights=empty_tensor,
            node_degrees=node_degrees,
            csr_indices=csr_indices,
            csr_offsets=csr_offsets,
            cached_src=empty_tensor,
            cached_dst=empty_tensor,
            cached_weights=empty_tensor,
            cache_indices=empty_tensor,
            fanouts=fanouts,
            alpha=2.0,
            use_cache=False,
            max_fanout=1024
        )
    
    end_time = time.time()
    
    sampled_neighbors = result[0]
    neighbor_counts = result[1]
    
    print(f"✓ Sampling completed in {end_time - start_time:.4f} seconds")
    print(f"✓ Sampled neighbors shape: {sampled_neighbors.shape}")
    print(f"✓ Neighbor counts shape: {neighbor_counts.shape}")
    print(f"✓ Average neighbors per seed: {neighbor_counts.float().mean():.2f}")
    print(f"✓ Total neighbors sampled: {neighbor_counts.sum().item()}")

def example_graph_filtering():
    """Example of graph structure masking and filtering"""
    print("\n" + "="*60)
    print("Example 2: Graph Structure Masking and Filtering")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sample graph
    src_nodes, dst_nodes, node_degrees, csr_indices, csr_offsets = create_sample_graph(
        num_nodes=1000, num_edges=5000, device=device
    )
    
    # Create degree thresholds (filter out low-degree nodes)
    degree_thresholds = torch.full((1000,), 5, device=device)  # Keep nodes with degree > 5
    
    # Create edge masks (randomly mask 50% of edges)
    edge_masks = torch.bernoulli(torch.full((src_nodes.size(0),), 0.5), device=device)
    
    print("Running graph structure masking...")
    start_time = time.time()
    
    if CUDA_AVAILABLE:
        result = graphsnapshot_cuda.graph_structure_mask(
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            edge_weights=torch.empty(0),
            node_degrees=node_degrees,
            csr_indices=csr_indices,
            csr_offsets=csr_offsets,
            node_degree_thresholds=degree_thresholds,
            edge_masks=edge_masks
        )
    else:
        result = graphsnapshot_cpu.graph_structure_mask(
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            edge_weights=torch.empty(0),
            node_degrees=node_degrees,
            csr_indices=csr_indices,
            csr_offsets=csr_offsets,
            node_degree_thresholds=degree_thresholds,
            edge_masks=edge_masks
        )
    
    end_time = time.time()
    
    output_src, output_dst, output_weights, valid_nodes, valid_edges, node_mapping, edge_mapping = result
    
    print(f"✓ Filtering completed in {end_time - start_time:.4f} seconds")
    print(f"✓ Original graph: {src_nodes.size(0)} edges")
    print(f"✓ Filtered graph: {valid_edges.item()} valid edges")
    print(f"✓ Valid nodes: {valid_nodes.item()}")
    print(f"✓ Compression ratio: {(1 - valid_edges.item() / src_nodes.size(0)) * 100:.1f}%")

def example_multi_hop_sampling():
    """Example of multi-hop neighbor aggregation"""
    print("\n" + "="*60)
    print("Example 3: Multi-hop Neighbor Aggregation")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sample graph
    src_nodes, dst_nodes, node_degrees, csr_indices, csr_offsets = create_sample_graph(
        num_nodes=1000, num_edges=5000, device=device
    )
    
    # Create seed nodes
    num_seeds = 50
    seed_nodes = torch.randint(0, 1000, (num_seeds,), device=device)
    
    # Define fanouts for each hop
    num_hops = 3
    hop_fanouts = torch.tensor([5, 3, 2], device=device)  # 5 neighbors in hop 1, 3 in hop 2, 2 in hop 3
    
    print("Running multi-hop neighbor aggregation...")
    start_time = time.time()
    
    if CUDA_AVAILABLE:
        result = graphsnapshot_cuda.multi_hop_aggregation(
            seed_nodes=seed_nodes,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            node_degrees=node_degrees,
            csr_indices=csr_indices,
            csr_offsets=csr_offsets,
            hop_fanouts=hop_fanouts,
            num_hops=num_hops,
            max_fanout=1024
        )
    else:
        result = graphsnapshot_cpu.multi_hop_aggregation(
            seed_nodes=seed_nodes,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            node_degrees=node_degrees,
            csr_indices=csr_indices,
            csr_offsets=csr_offsets,
            hop_fanouts=hop_fanouts,
            num_hops=num_hops,
            max_fanout=1024
        )
    
    end_time = time.time()
    
    hop_neighbors, hop_counts, visited_mask = result
    
    print(f"✓ Multi-hop aggregation completed in {end_time - start_time:.4f} seconds")
    print(f"✓ Hop neighbors shape: {hop_neighbors.shape}")
    print(f"✓ Hop counts shape: {hop_counts.shape}")
    print(f"✓ Visited mask shape: {visited_mask.shape}")
    
    for hop in range(num_hops):
        avg_neighbors = hop_counts[:, hop].float().mean()
        print(f"✓ Hop {hop + 1}: Average {avg_neighbors:.2f} neighbors per seed")

def example_performance_comparison():
    """Compare performance between CUDA and CPU implementations"""
    print("\n" + "="*60)
    print("Example 4: Performance Comparison")
    print("="*60)
    
    if not CUDA_AVAILABLE:
        print("⚠ CUDA not available, skipping performance comparison")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create larger graph for performance testing
    src_nodes, dst_nodes, node_degrees, csr_indices, csr_offsets = create_sample_graph(
        num_nodes=10000, num_edges=50000, device=device
    )
    
    # Create seed nodes
    num_seeds = 1000
    seed_nodes = torch.randint(0, 10000, (num_seeds,), device=device)
    fanouts = torch.full((num_seeds,), 20, device=device)
    
    empty_tensor = torch.empty(0, device=device)
    
    # Test CUDA performance
    print("Testing CUDA performance...")
    cuda_times = []
    for i in range(5):  # Run 5 iterations
        torch.cuda.synchronize()
        start_time = time.time()
        
        graphsnapshot_cuda.neighbor_sampling_fcr(
            seed_nodes=seed_nodes,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            edge_weights=empty_tensor,
            node_degrees=node_degrees,
            csr_indices=csr_indices,
            csr_offsets=csr_offsets,
            cached_src=empty_tensor,
            cached_dst=empty_tensor,
            cached_weights=empty_tensor,
            cache_indices=empty_tensor,
            fanouts=fanouts,
            alpha=2.0,
            use_cache=False,
            max_fanout=1024
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        cuda_times.append(end_time - start_time)
    
    # Test CPU performance
    print("Testing CPU performance...")
    cpu_times = []
    for i in range(5):  # Run 5 iterations
        start_time = time.time()
        
        graphsnapshot_cpu.neighbor_sampling_fcr(
            seed_nodes=seed_nodes.cpu(),
            src_nodes=src_nodes.cpu(),
            dst_nodes=dst_nodes.cpu(),
            edge_weights=empty_tensor.cpu(),
            node_degrees=node_degrees.cpu(),
            csr_indices=csr_indices.cpu(),
            csr_offsets=csr_offsets.cpu(),
            cached_src=empty_tensor.cpu(),
            cached_dst=empty_tensor.cpu(),
            cached_weights=empty_tensor.cpu(),
            cache_indices=empty_tensor.cpu(),
            fanouts=fanouts.cpu(),
            alpha=2.0,
            use_cache=False,
            max_fanout=1024
        )
        
        end_time = time.time()
        cpu_times.append(end_time - start_time)
    
    # Calculate averages
    avg_cuda_time = np.mean(cuda_times)
    avg_cpu_time = np.mean(cpu_times)
    speedup = avg_cpu_time / avg_cuda_time
    
    print(f"✓ Average CUDA time: {avg_cuda_time:.4f} seconds")
    print(f"✓ Average CPU time: {avg_cpu_time:.4f} seconds")
    print(f"✓ CUDA speedup: {speedup:.2f}x")

def main():
    """Main function to run all examples"""
    print("GraphSnapShot CUDA Kernels Example Usage")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    try:
        # Run examples
        example_fcr_sampling()
        example_graph_filtering()
        example_multi_hop_sampling()
        example_performance_comparison()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
