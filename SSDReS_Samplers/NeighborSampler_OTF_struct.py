import dgl
from .. import backend as F
from ..base import EID, NID
from ..heterograph import DGLGraph
from ..transforms import to_block
from ..utils import get_num_threads
from .base import BlockSampler
import torch

def graph_difference(g1, g2):
    """
    Computes the set difference of edges between two DGL graphs g1 and g2. 
    It returns a new graph containing only the edges that are present in g1 
    but not in g2. This function is useful for operations like cache refreshing 
    in graph sampling where unique edges need to be identified.

    Parameters
    ----------
    g1 : DGLGraph
        The first graph from which edges are to be subtracted.
    g2 : DGLGraph
        The second graph whose edges are to be considered for subtraction from g1.

    Returns
    -------
    DGLGraph
        A new DGLGraph containing only the edges that are in g1 but not in g2.

    """
    # Convert edges to sets of tuples for easy comparison
    edges_g1 = set(zip(*g1.edges()))
    edges_g2 = set(zip(*g2.edges()))
    
    # Find edges present in g1 but not in g2
    unique_edges = edges_g1 - edges_g2
    unique_src, unique_dst = zip(*unique_edges) if unique_edges else ([], [])
    
    # Create a new graph from the unique edges
    g_unique = dgl.graph((torch.tensor(unique_src), torch.tensor(unique_dst)), num_nodes=g1.number_of_nodes())
    return g_unique

class NeighborSampler_OTF_struct:
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    Parameters
    ----------
    g : DGLGraph
        The input graph from which neighbors are sampled.
    fanouts : list[int]
        The number of neighbors to sample for each layer of the GNN.
    edge_dir : str, default 'in'
        The direction of edge sampling ('in' or 'out').
    alpha, beta, gamma : float
        Parameters controlling the cache amplification (alpha), cache refresh rate (beta), 
        and the proportion of the cache to refresh (gamma).
    prob : Tensor, optional
        Edge weights to be used as probabilities for sampling.
    replace : bool, default False
        Whether to sample with replacement.
    output_device : torch.device, optional
        The device on which the sampled blocks will be placed.
    exclude_eids : Tensor, optional
        Edges to exclude from sampling.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> import numpy as np
    >>> 
    >>> # Create a random graph with 100 nodes and 300 directed edges
    >>> num_nodes = 100
    >>> num_edges = 300
    >>> edges_src = np.random.randint(0, num_nodes, num_edges)
    >>> edges_dst = np.random.randint(0, num_nodes, num_edges)
    >>> g = dgl.graph((edges_src, edges_dst))
    >>> g.ndata['feat'] = torch.randn(num_nodes, 10)  # Assign random features to nodes
    >>>
    >>> # Initialize the sampler with specified fanouts and cache parameters
    >>> fanouts = [5, 10]  # Define fanouts for two GNN layers
    >>> sampler = NeighborSampler_OTF_struct(g, fanouts, edge_dir='in', alpha=0.6, beta=2, gamma=0.4)
    >>>
    >>> # Perform sampling for a batch of seed nodes
    >>> seed_nodes = torch.tensor([0, 1, 2, 3, 4])  # Define seed nodes
    >>> new_seed_nodes, blocks = sampler.sample_blocks(seed_nodes)
    >>>
    >>> # Display information about the sampled blocks
    >>> for i, block in enumerate(blocks):
    ...     print(f"Layer {i}:")
    ...     print("  Number of source nodes:", block.num_src_nodes())
    ...     print("  Number of destination nodes:", block.num_dst_nodes())
    ...     print("  Number of edges:", block.num_edges())

    This example demonstrates initializing the `NeighborSampler_OTF_struct` with a DGL graph, specifying the fanouts
    for sampling, and then performing neighbor sampling. The sampled blocks can be used for constructing GNN layers.
    """
    
    def __init__(self, g, fanouts, edge_dir='in', alpha=0.6, beta=2, gamma=0.4, prob=None, replace=False, output_device=None, exclude_eids=None):
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prob = prob
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids
        
        # Initialize cache with amplified fanouts
        self.cached_graph_structures = [self.initialize_cache(fanout * alpha * beta) for fanout in fanouts]

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes(), dtype=torch.int64),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        return cached_graph

    def refresh_cache(self, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        # Sample edges to remove from cache
        to_remove = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        
        # Compute the difference and update the cache
        removed = graph_difference(cached_graph_structure, to_remove)
        
        # Add new edges from the disk to the cache
        to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        
        # Merge the updated cache with new samples
        refreshed_cache = dgl.graph((torch.cat([removed.edges()[0], to_add.edges()[0]]),
                                     torch.cat([removed.edges()[1], to_add.edges()[1]])),
                                    num_nodes=self.g.number_of_nodes())
        return refreshed_cache

    def sample_blocks(self, seed_nodes):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        for i, (fanout, cached_graph_structure) in enumerate(zip(self.fanouts, self.cached_graph_structures)):
            fanout_cache_retrieval = int(fanout * self.alpha)
            fanout_disk = fanout - fanout_cache_retrieval
            fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)
            
            # Refresh cache partially
            self.cached_graph_structures[i] = self.refresh_cache(cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            
            # Sample from cache
            frontier_cache = self.cached_graph_structures[i].sample_neighbors(
                seed_nodes,
                fanout_cache_retrieval,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            
            # Sample remaining from disk
            frontier_disk = self.g.sample_neighbors(
                seed_nodes,
                fanout_disk,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            
            # Merge frontiers
            merged_frontier = dgl.mege([frontier_cache, frontier_disk]) #merge batch
            
            # Convert the merged frontier to a block
            block = to_block(self.g, merged_frontier, seed_nodes)
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

            print(f"Layer {i}: Merged frontier edges:", merged_frontier.edges())

        return seed_nodes, blocks
