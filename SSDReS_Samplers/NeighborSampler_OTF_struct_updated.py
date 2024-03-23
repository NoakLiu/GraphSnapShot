from .. import backend as F
from ..base import EID, NID
from ..heterograph import DGLGraph
from ..transforms import to_block
from ..utils import get_num_threads
from .base import BlockSampler
import torch
import dgl

class NeighborSampler_OTF_struct_rc(BlockSampler):
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
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                alpha=0.6, 
                beta=2, 
                gamma=0.4, 
                T = 50,
                prob=None, 
                replace=False, 
                output_device=None, 
                exclude_eids=None,
                mask=None,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                fused=True,
                 ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids
        self.T = T
        self.cycle = 0
        self.Toptim = int(self.g.number_of_nodes()/max(self.fanouts))

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.cache_size = [fanout * alpha * beta for fanout in fanouts]
        # print(self.cache_size)

        # Initialize cache with amplified fanouts
        self.cached_graph_structures = [self.initialize_cache(fanout * alpha * beta) for fanout in fanouts]

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        # print("begin init")
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes(), dtype=torch.int64),
            torch.arange(0, self.g.number_of_nodes()),
            # self.g.number_of_nodes(),
            # 10,
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end init")
        return cached_graph

    def refresh_cache(self,layer_id, cached_graph_structure,fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample=[]
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        # Sample edges to remove from cache
        cache_remain_storage = cached_graph_structure.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        disk_to_add_storage = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([cache_remain_storage, disk_to_add_storage])
        refreshed_cache = dgl.add_self_loop(refreshed_cache)
        return refreshed_cache


    def cache_retrieval(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_retrieval = min(fanout_cache_retrieval, self.cache_size[layer_id])
        cache_remain_compute = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_retrieval,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        disk_to_add_compute = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([cache_remain_compute, disk_to_add_compute])
        refreshed_cache = dgl.add_self_loop(refreshed_cache)
        return refreshed_cache


    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        # print("in sample blocks")
        for i, (fanout, cached_graph_structure) in enumerate(zip(reversed(self.fanouts), reversed(self.cached_graph_structures))):
            self.cycle+=1
            if (self.cycle%self.Toptim==0):
                fanout_cache_retrieval = int(fanout * self.alpha)
                fanout_disk = fanout - fanout_cache_retrieval
                fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)

                # print("fanout_size:",fanout)

                # Refresh cache partially
                # self.cached_graph_structures[i] 
                self.cached_graph_structures[i]= self.refresh_cache(i, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            
            if (self.cycle%self.T==0):
                frontier_cache = self.cache_retrieval(i, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            else:
                # Sample from cache
                frontier_cache = self.cached_graph_structures[i].sample_neighbors(
                    seed_nodes,
                    #fanout_cache_retrieval,
                    fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=self.exclude_eids,
                )
            
            # Convert the merged frontier to a block
            block = to_block(frontier_cache, seed_nodes)
            if EID in frontier_cache.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_cache.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks