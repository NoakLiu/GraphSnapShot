"""Data loading components for neighbor sampling"""
from .. import backend as F
from ..base import EID, NID
from ..heterograph import DGLGraph
from ..transforms import to_block
from ..utils import get_num_threads
from .base import BlockSampler

import torch
import dgl


class NeighborSampler(BlockSampler):
    """Sampler that builds computational dependency of node representations via
    neighbor sampling for multilayer GNN.

    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type.  The neighbors are picked uniformly.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.

        If only a single integer is provided, DGL assumes that every edge type
        will have the same fanout.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    edge_dir : str, default ``'in'``
        Can be either ``'in' `` where the neighbors will be sampled according to
        incoming edges, or ``'out'`` otherwise, same as :func:`dgl.sampling.sample_neighbors`.
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``.  The feature must be
        a scalar on each edge.

        This argument is mutually exclusive with :attr:`mask`.  If you want to
        specify both a mask and a probability, consider multiplying the probability
        with the mask instead.
    mask : str, optional
        If given, a neighbor could be picked only if the edge mask with the given
        name in ``g.edata`` is True.  The data must be boolean on each edge.

        This argument is mutually exclusive with :attr:`prob`.  If you want to
        specify both a mask and a probability, consider multiplying the probability
        with the mask instead.
    replace : bool, default False
        Whether to sample with replacement
    prefetch_node_feats : list[str] or dict[ntype, list[str]], optional
        The source node data to prefetch for the first MFG, corresponding to the
        input node features necessary for the first GNN layer.
    prefetch_labels : list[str] or dict[ntype, list[str]], optional
        The destination node data to prefetch for the last MFG, corresponding to
        the node labels of the minibatch.
    prefetch_edge_feats : list[str] or dict[etype, list[str]], optional
        The edge data names to prefetch for all the MFGs, corresponding to the
        edge features necessary for all GNN layers.
    output_device : device, optional
        The device of the output subgraphs or MFGs.  Default is the same as the
        minibatch of seed nodes.
    fused : bool, default True
        If True and device is CPU fused sample neighbors is invoked. This version
        requires seed_nodes to be unique

    Examples
    --------
    **Node classification**

    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15])
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.NeighborSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)

    If you would like non-uniform neighbor sampling:

    >>> g.edata['p'] = torch.rand(g.num_edges())   # any non-negative 1D vector works
    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15], prob='p')

    Or sampling on edge masks:

    >>> g.edata['mask'] = torch.rand(g.num_edges()) < 0.2   # any 1D boolean mask works
    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15], prob='mask')

    **Edge classification and link prediction**

    This class can also work for edge classification and link prediction together
    with :func:`as_edge_prediction_sampler`.

    >>> sampler = dgl.dataloading.NeighborSampler([5, 10, 15])
    >>> sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_eid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)

    See the documentation :func:`as_edge_prediction_sampler` for more details.

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                print("hiiiiiiii is dict")
                for ntype in list(seed_nodes.keys()):
                    print("seed dict",seed_nodes.keys)
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                print("hiiiiiiii is dglgraphobject")
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for fanout in reversed(self.fanouts):
            print("seeds nodes:",seed_nodes)
            print("org g:",g)
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            print("sampled frontier:",frontier)
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks


MultiLayerNeighborSampler = NeighborSampler


class MultiLayerFullNeighborSampler(NeighborSampler):
    """Sampler that builds computational dependency of node representations by taking messages
    from all neighbors for multilayer GNN.

    This sampler will make every node gather messages from every single neighbor per edge type.

    Parameters
    ----------
    num_layers : int
        The number of GNN layers to sample.
    kwargs :
        Passed to :class:`dgl.dataloading.NeighborSampler`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors for the first,
    second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(self, num_layers, **kwargs):
        super().__init__([-1] * num_layers, **kwargs)


NeighborSampler_FBL = NeighborSampler

class MultiLayerFullNeighborSampler(NeighborSampler):
    """Sampler that builds computational dependency of node representations by taking messages
    from all neighbors for multilayer GNN.

    This sampler will make every node gather messages from every single neighbor per edge type.

    Parameters
    ----------
    num_layers : int
        The number of GNN layers to sample.
    kwargs :
        Passed to :class:`dgl.dataloading.NeighborSampler`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors for the first,
    second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """

    def __init__(self, num_layers, **kwargs):
        super().__init__([-1] * num_layers, **kwargs)

class NeighborSampler_OTF_struct(BlockSampler):
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
                alpha=0.6, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                beta=2, # 
                gamma=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=50, # refresh gap
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
        self.T = T
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
            mappings=self.mapping
        )
        # print("end init")
        return cached_graph

    def refresh_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        # print("begin refresh")
        # print("begin remove")
        fanout_cache_sample=[]
        # print("layer_id",layer_id)
        # print("self.cache.size",self.cache_size)
        # print("fanout_cache_refresh",fanout_cache_refresh)
        # for i in range(len(self.cache_size)):
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        # print(fanout_cache_sample)
        # Sample edges to remove from cache
        removed = cached_graph_structure.sample_neighbors(
            seed_nodes,
            #fanout_cache_refresh,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end remove")
        
        # # Compute the difference and update the cache
        # print("removed")
        # # removed = graph_difference(cached_graph_structure, to_remove)
        # print("end removed")
        
        # Add new edges from the disk to the cache
        # print("add graph")
        to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end add graph")
        
        # # Merge the updated cache with new samples
        # print("begin refresh cache")
        # refreshed_cache = dgl.graph((torch.cat([removed.edges()[0], to_add.edges()[0]]),
        #                              torch.cat([removed.edges()[1], to_add.edges()[1]])),
        #                             num_nodes=self.g.number_of_nodes())
        refreshed_cache = dgl.merge([removed, to_add])
        print("end refresh cache")
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
            fanout_cache_retrieval = int(fanout * self.alpha)
            fanout_disk = fanout - fanout_cache_retrieval
            fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)

            # print("fanout_size:",fanout)

            # Refresh cache partially
            self.cached_graph_structures[i] = self.refresh_cache(i, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            
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
            # print("merged_cache",frontier_cache)

            merged_frontier = frontier_cache
            
            # # Sample remaining from disk
            # frontier_disk = g.sample_neighbors(
            #     seed_nodes,
            #     fanout_disk,
            #     edge_dir=self.edge_dir,
            #     prob=self.prob,
            #     replace=self.replace,
            #     output_device=self.output_device,
            #     exclude_edges=self.exclude_eids,
            # )
            # print("frontier_disk",frontier_disk)
            
            # # Merge frontiers
            # merged_frontier = dgl.merge([frontier_cache, frontier_disk]) #merge batch
            
            # Convert the merged frontier to a block
            block = to_block(merged_frontier, seed_nodes)
            if EID in merged_frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = merged_frontier.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

            # print(f"Layer {i}: Merged frontier edges:", merged_frontier.edges())
        

        return seed_nodes,output_nodes, blocks


class NeighborSampler_FCR_struct(BlockSampler):
    """
    A neighbor sampler that supports cache-refreshing (FCR) for efficient sampling, 
    tailored for multi-layer GNNs. This sampler augments the sampling process by 
    maintaining a cache of pre-sampled neighborhoods that can be reused across 
    multiple sampling iterations. It introduces cache amplification (via the alpha 
    parameter) and cache refresh cycles (via the T parameter) to manage the balance 
    between sampling efficiency and freshness of the sampled neighborhoods.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.
    edge_dir : str, default "in"
        Direction of sampling. Can be either "in" for incoming edges or "out" for outgoing edges.
    prob : str, optional
        Name of the edge feature in g.edata used as the probability for edge sampling.
    alpha : int, default 2
        Cache amplification ratio. Determines the size of the pre-sampled cache relative
        to the actual sampling needs. A larger alpha means more neighbors are pre-sampled.
    T : int, default 1
        Cache refresh cycle. Specifies how often (in terms of sampling iterations) the
        cache should be refreshed.

    Examples
    --------
    Initialize a graph and a NeighborSampler_FCR_struct for a 2-layer GNN with fanouts
    [5, 10]. Assume alpha=2 for double the size of pre-sampling and T=3 for refreshing
    the cache every 3 iterations.

    >>> import dgl
    >>> import torch
    >>> g = dgl.rand_graph(100, 200)  # Random graph with 100 nodes and 200 edges
    >>> g.ndata['feat'] = torch.randn(100, 10)  # Random node features
    >>> sampler = NeighborSampler_FCR_struct(g, [5, 10], alpha=2, T=3)
    
    To perform sampling:

    >>> seed_nodes = torch.tensor([1, 2, 3])  # Nodes for which neighbors are sampled
    >>> for i in range(5):  # Simulate 5 sampling iterations
    ...     seed_nodes, output_nodes, blocks = sampler.sample_blocks(seed_nodes)
    ...     # Process the sampled blocks
    """
    
    def __init__(
            self, 
            g, 
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=1,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):
        self.g = g

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.T = T
        self.cycle = 0  # Initialize sampling cycle counter
        self.amplified_fanouts = [f * alpha for f in fanouts]  # Amplified fanouts for pre-sampling
        self.cache_struct = []  # Initialize cache structure
        self.cache_refresh()  # Pre-sample and populate the cache

    def cache_refresh(self,exclude_eids=None):
        """
        Pre-samples neighborhoods with amplified fanouts and refreshes the cache. This method
        is automatically called upon initialization and after every T sampling iterations to
        ensure that the cache is periodically updated with fresh samples.
        """
        self.cache_struct.clear()  # Clear existing cache
        for fanout in self.amplified_fanouts:
            # Sample neighbors for each layer with amplified fanout
            # print("large")
            # print(fanout)
            # print("---")
            frontier = self.g.sample_neighbors(
                torch.arange(0, self.g.number_of_nodes()),  # Consider all nodes as seeds for pre-sampling
                # self.g.number_of_nodes(),
                # 10,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )
            frontier = dgl.add_self_loop(frontier)
            # print(frontier)
            # print(self.cache_struct)
            # print("then append")
            self.cache_struct.append(frontier)  # Update cache with new samples

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes

        self.cycle += 1
        if self.cycle % self.T == 0:
            self.cache_refresh()  # Refresh cache every T cycles

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        k = 0
        # print("cache struct")
        # print(self.cache_struct)
        # print(len(self.cache_struct))
        # print("---")
        for k in range(len(self.cache_struct)-1,-1,-1):
            frontier_large = self.cache_struct[k]
            fanout = self.fanouts[k]
            # print("small")
            # print("seed",seed_nodes)
            # print("fanout",fanout)
            # print("frontier_large",frontier_large)
            # print("---")
            frontier = frontier_large.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # frontier = g.sample_neighbors(
            #     # torch.arange(0, self.g.number_of_nodes()),  # Consider all nodes as seeds for pre-sampling
            #     # self.g.number_of_nodes(),
            #     seed_nodes,
            #     fanout,
            #     edge_dir=self.edge_dir,
            #     prob=self.prob,
            #     replace=self.replace,
            #     output_device=self.output_device,
            #     exclude_edges=exclude_eids
            # )

            # Directly use pre-sampled frontier from the cache
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks
    

class NeighborSampler_FCR_struct_hete(BlockSampler):
    """
    A neighbor sampler that supports cache-refreshing (FCR) for efficient sampling, 
    tailored for multi-layer GNNs. This sampler augments the sampling process by 
    maintaining a cache of pre-sampled neighborhoods that can be reused across 
    multiple sampling iterations. It introduces cache amplification (via the alpha 
    parameter) and cache refresh cycles (via the T parameter) to manage the balance 
    between sampling efficiency and freshness of the sampled neighborhoods.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.
    edge_dir : str, default "in"
        Direction of sampling. Can be either "in" for incoming edges or "out" for outgoing edges.
    prob : str, optional
        Name of the edge feature in g.edata used as the probability for edge sampling.
    alpha : int, default 2
        Cache amplification ratio. Determines the size of the pre-sampled cache relative
        to the actual sampling needs. A larger alpha means more neighbors are pre-sampled.
    T : int, default 1
        Cache refresh cycle. Specifies how often (in terms of sampling iterations) the
        cache should be refreshed.

    Examples
    --------
    Initialize a graph and a NeighborSampler_FCR_struct for a 2-layer GNN with fanouts
    [5, 10]. Assume alpha=2 for double the size of pre-sampling and T=3 for refreshing
    the cache every 3 iterations.

    >>> import dgl
    >>> import torch
    >>> g = dgl.rand_graph(100, 200)  # Random graph with 100 nodes and 200 edges
    >>> g.ndata['feat'] = torch.randn(100, 10)  # Random node features
    >>> sampler = NeighborSampler_FCR_struct(g, [5, 10], alpha=2, T=3)
    
    To perform sampling:

    >>> seed_nodes = torch.tensor([1, 2, 3])  # Nodes for which neighbors are sampled
    >>> for i in range(5):  # Simulate 5 sampling iterations
    ...     seed_nodes, output_nodes, blocks = sampler.sample_blocks(seed_nodes)
    ...     # Process the sampled blocks
    """
    
    def __init__(
            self, 
            g, 
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=1,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):
        self.g = g

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.T = T
        self.cycle = 0  # Initialize sampling cycle counter
        self.amplified_fanouts = [f * alpha for f in fanouts]  # Amplified fanouts for pre-sampling
        self.cache_struct = []  # Initialize cache structure
        # self.cache_refresh()  # Pre-sample and populate the cache

    def cache_refresh(self,exclude_eids=None):
        """
        Pre-samples neighborhoods with amplified fanouts and refreshes the cache. This method
        is automatically called upon initialization and after every T sampling iterations to
        ensure that the cache is periodically updated with fresh samples.
        """
        self.cache_struct.clear()  # Clear existing cache
        for fanout in self.amplified_fanouts:
            # Sample neighbors for each layer with amplified fanout
            # print("large")
            # print(fanout)
            # print("---")
            # sample_seeds = {'paper':[torch.arange(0, self.g.num_nodes("paper"))]}

            # sample_seeds = self.g.nodes
            # print(sample_seeds)
            # print("here is sample seeds",sample_seeds)
            # pnum = list(range(0,100))
            # print("---for seed nodes---",seed_nodes)
            frontier = self.g.sample_neighbors(
                # {'paper':list(seed_nodes['paper'])},
                {'paper':list(range(0, self.g.num_nodes("paper")))},
                #  {'paper':pnum, 'author':pnum},
                # sample_seeds,  # Consider all nodes as seeds for pre-sampling
                # self.g.number_of_nodes(),
                # 10,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
                # ntype='paper'
                # copy_ndata = True,
                # copy_edata = True,
            )
            # frontier = dgl.add_self_loop(frontier)
            # print(frontier)
            # print(self.cache_struct)
            # print("then append")
            self.cache_struct.append(frontier)  # Update cache with new samples

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        print("sample seed nodes",seed_nodes)
        output_nodes = seed_nodes

        # self.cycle += 1
        if self.cycle % self.T == 0:
            self.cache_refresh(seed_nodes)  # Refresh cache every T cycles
        self.cycle += 1

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        k = 0
        # print("cache struct")
        # print(self.cache_struct)
        # print(len(self.cache_struct))
        # print("---")
        for k in range(len(self.cache_struct)-1,-1,-1):
            # frontier_large = self.cache_struct[k]
            fanout = self.fanouts[k]
            # print("small")
            # print("seed",seed_nodes)
            # print("fanout",fanout)
            # print("frontier_large",frontier_large)
            # print("---")
            frontier = self.cache_struct[k].sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # frontier = g.sample_neighbors(
            #     # torch.arange(0, self.g.number_of_nodes()),  # Consider all nodes as seeds for pre-sampling
            #     # self.g.number_of_nodes(),
            #     seed_nodes,
            #     fanout,
            #     edge_dir=self.edge_dir,
            #     prob=self.prob,
            #     replace=self.replace,
            #     output_device=self.output_device,
            #     exclude_edges=exclude_eids
            # )

            # Directly use pre-sampled frontier from the cache
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        
        print("----------------For each element------------")
        print("-----seed_nodes-----",seed_nodes)
        print("-----output_nodes-----",output_nodes)
        print("-----blocks-----",blocks)
        
        return seed_nodes, output_nodes, blocks
    

class NeighborSampler_FCR_struct_shared_cache(BlockSampler):
    """
    A neighbor sampler that supports cache-refreshing (FCR) for efficient sampling, 
    tailored for multi-layer GNNs. This sampler augments the sampling process by 
    maintaining a cache of pre-sampled neighborhoods that can be reused across 
    multiple sampling iterations. It introduces cache amplification (via the alpha 
    parameter) and cache refresh cycles (via the T parameter) to manage the balance 
    between sampling efficiency and freshness of the sampled neighborhoods.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.
    edge_dir : str, default "in"
        Direction of sampling. Can be either "in" for incoming edges or "out" for outgoing edges.
    prob : str, optional
        Name of the edge feature in g.edata used as the probability for edge sampling.
    alpha : int, default 2
        Cache amplification ratio. Determines the size of the pre-sampled cache relative
        to the actual sampling needs. A larger alpha means more neighbors are pre-sampled.
    T : int, default 1
        Cache refresh cycle. Specifies how often (in terms of sampling iterations) the
        cache should be refreshed.

    Examples
    --------
    Initialize a graph and a NeighborSampler_FCR_struct for a 2-layer GNN with fanouts
    [5, 10]. Assume alpha=2 for double the size of pre-sampling and T=3 for refreshing
    the cache every 3 iterations.

    >>> import dgl
    >>> import torch
    >>> g = dgl.rand_graph(100, 200)  # Random graph with 100 nodes and 200 edges
    >>> g.ndata['feat'] = torch.randn(100, 10)  # Random node features
    >>> sampler = NeighborSampler_FCR_struct(g, [5, 10], alpha=2, T=3)
    
    To perform sampling:

    >>> seed_nodes = torch.tensor([1, 2, 3])  # Nodes for which neighbors are sampled
    >>> for i in range(5):  # Simulate 5 sampling iterations
    ...     seed_nodes, output_nodes, blocks = sampler.sample_blocks(seed_nodes)
    ...     # Process the sampled blocks
    """
    
    def __init__(
            self, 
            g, 
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=1,
            prob=None,
            mask=None,
            replace=False,
            prefetch_node_feats=None,
            prefetch_labels=None,
            prefetch_edge_feats=None,
            output_device=None,
            fused=True,
        ):
        self.g = g

        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}

        self.alpha = alpha
        self.T = T
        self.cycle = 0  # Initialize sampling cycle counter
        self.amplified_fanout = max([f * alpha for f in fanouts])  # Amplified fanouts for pre-sampling
        self.cache_struct = []  # Initialize cache structure
        self.cache_refresh()  # Pre-sample and populate the cache

    def cache_refresh(self,exclude_eids=None):
        """
        Pre-samples neighborhoods with amplified fanouts and refreshes the cache. This method
        is automatically called upon initialization and after every T sampling iterations to
        ensure that the cache is periodically updated with fresh samples.
        """
        # self.cache_struct.clear()  # Clear existing cache
        # Sample neighbors for each layer with amplified fanout
        # print("large")
        # # print(fanout)
        # print("---")
        frontier = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),  # Consider all nodes as seeds for pre-sampling
            # self.g.number_of_nodes(),
            # 10,
            self.amplified_fanout,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )
        frontier = dgl.add_self_loop(frontier)
        # print(frontier)
        # print(self.cache_struct)
        # print("then append")
        self.cache_struct=frontier  # Update cache with new samples

    def sample_blocks(self, g,seed_nodes, exclude_eids=None):
        """
        Samples blocks from the graph for the specified seed nodes using the cache.

        Parameters
        ----------
        seed_nodes : Tensor
            The nodes for which the neighborhoods are to be sampled.

        Returns
        -------
        tuple
            A tuple containing the seed nodes for the next layer, the output nodes, and
            the list of blocks sampled from the graph.
        """
        output_nodes = seed_nodes
        self.cycle += 1
        if self.cycle % self.T == 0:
            self.cache_refresh()  # Refresh cache every T cycles

        blocks = []

        if self.fused and get_num_threads() > 1:
            # print("fused")
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        k = 0
        # print("cache struct")
        # print(self.cache_struct)
        # print(len(self.cache_struct))
        # print("---")
        for k in range(len(self.fanouts)-1,-1,-1):
            frontier_large = self.cache_struct
            fanout = self.fanouts[k]
            # print("small")
            # print("seed",seed_nodes)
            # print("fanout",fanout)
            # print("frontier_large",frontier_large)
            # print("---")
            frontier = frontier_large.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # frontier = g.sample_neighbors(
            #     # torch.arange(0, self.g.number_of_nodes()),  # Consider all nodes as seeds for pre-sampling
            #     # self.g.number_of_nodes(),
            #     seed_nodes,
            #     fanout,
            #     edge_dir=self.edge_dir,
            #     prob=self.prob,
            #     replace=self.replace,
            #     output_device=self.output_device,
            #     exclude_edges=exclude_eids
            # )

            # Directly use pre-sampled frontier from the cache
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        
        return seed_nodes, output_nodes, blocks
    

class NeighborSampler_OTF_struct(BlockSampler):
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

    def refresh_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        # print("begin refresh")
        # print("begin remove")
        fanout_cache_sample=[]
        # print("layer_id",layer_id)
        # print("self.cache.size",self.cache_size)
        # print("fanout_cache_refresh",fanout_cache_refresh)
        # for i in range(len(self.cache_size)):
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        # print(fanout_cache_sample)
        # Sample edges to remove from cache
        removed = cached_graph_structure.sample_neighbors(
            seed_nodes,
            #fanout_cache_refresh,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end remove")
        
        # # Compute the difference and update the cache
        # print("removed")
        # # removed = graph_difference(cached_graph_structure, to_remove)
        # print("end removed")
        
        # Add new edges from the disk to the cache
        # print("add graph")
        to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end add graph")
        
        # # Merge the updated cache with new samples
        # print("begin refresh cache")
        # refreshed_cache = dgl.graph((torch.cat([removed.edges()[0], to_add.edges()[0]]),
        #                              torch.cat([removed.edges()[1], to_add.edges()[1]])),
        #                             num_nodes=self.g.number_of_nodes())
        refreshed_cache = dgl.merge([removed, to_add])
        refreshed_cache = dgl.add_self_loop(refreshed_cache)
        # print("end refresh cache")
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
            if (self.cycle%self.T==0):
                fanout_cache_retrieval = int(fanout * self.alpha)
                fanout_disk = fanout - fanout_cache_retrieval
                fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)

                # print("fanout_size:",fanout)

                # Refresh cache partially
                self.cached_graph_structures[i] = self.refresh_cache(i, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            
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
            print("merged_cache",frontier_cache)

            merged_frontier = frontier_cache
            # merged_frontier = dgl.add_self_loop(merged_frontier)
            
            # # Sample remaining from disk
            # frontier_disk = g.sample_neighbors(
            #     seed_nodes,
            #     fanout_disk,
            #     edge_dir=self.edge_dir,
            #     prob=self.prob,
            #     replace=self.replace,
            #     output_device=self.output_device,
            #     exclude_edges=self.exclude_eids,
            # )
            # print("frontier_disk",frontier_disk)
            
            # # Merge frontiers
            # merged_frontier = dgl.merge([frontier_cache, frontier_disk]) #merge batch
            
            # Convert the merged frontier to a block
            block = to_block(merged_frontier, seed_nodes)
            if EID in merged_frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = merged_frontier.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

            # print(f"Layer {i}: Merged frontier edges:", merged_frontier.edges())
        

        return seed_nodes,output_nodes, blocks
    

class NeighborSampler_OTF_struct_hete(BlockSampler):
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

    def refresh_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        # print("begin refresh")
        # print("begin remove")
        fanout_cache_sample=[]
        # print("layer_id",layer_id)
        # print("self.cache.size",self.cache_size)
        # print("fanout_cache_refresh",fanout_cache_refresh)
        # for i in range(len(self.cache_size)):
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        # print(fanout_cache_sample)
        # Sample edges to remove from cache
        removed = cached_graph_structure.sample_neighbors(
            seed_nodes,
            #fanout_cache_refresh,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end remove")
        
        # # Compute the difference and update the cache
        # print("removed")
        # # removed = graph_difference(cached_graph_structure, to_remove)
        # print("end removed")
        
        # Add new edges from the disk to the cache
        # print("add graph")
        to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end add graph")
        
        # # Merge the updated cache with new samples
        # print("begin refresh cache")
        # refreshed_cache = dgl.graph((torch.cat([removed.edges()[0], to_add.edges()[0]]),
        #                              torch.cat([removed.edges()[1], to_add.edges()[1]])),
        #                             num_nodes=self.g.number_of_nodes())
        refreshed_cache = dgl.merge([removed, to_add])
        refreshed_cache = dgl.add_self_loop(refreshed_cache)
        # print("end refresh cache")
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
            if (self.cycle%self.T==0):
                fanout_cache_retrieval = int(fanout * self.alpha)
                fanout_disk = fanout - fanout_cache_retrieval
                fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)

                # print("fanout_size:",fanout)

                # Refresh cache partially
                self.cached_graph_structures[i] = self.refresh_cache(i, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            
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
            # print("merged_cache",frontier_cache)

            merged_frontier = frontier_cache
            # merged_frontier = dgl.add_self_loop(merged_frontier)
            
            # # Sample remaining from disk
            # frontier_disk = g.sample_neighbors(
            #     seed_nodes,
            #     fanout_disk,
            #     edge_dir=self.edge_dir,
            #     prob=self.prob,
            #     replace=self.replace,
            #     output_device=self.output_device,
            #     exclude_edges=self.exclude_eids,
            # )
            # print("frontier_disk",frontier_disk)
            
            # # Merge frontiers
            # merged_frontier = dgl.merge([frontier_cache, frontier_disk]) #merge batch
            
            # Convert the merged frontier to a block
            block = to_block(merged_frontier, seed_nodes)
            if EID in merged_frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = merged_frontier.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

            # print(f"Layer {i}: Merged frontier edges:", merged_frontier.edges())
        

        return seed_nodes,output_nodes, blocks


class NeighborSampler_OTF_struct_shared_cache(BlockSampler):
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
                T=50,
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

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.cache_size = max([fanout * alpha * beta for fanout in fanouts])
        # print(self.cache_size)

        # Initialize cache with amplified fanouts
        self.cached_graph_structure = self.initialize_cache(self.cache_size) 
        self.T = T
        self.cycle = 0

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

    def refresh_cache(self,cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        # print("begin refresh")
        # print("begin remove")
        fanout_cache_sample=[]
        # print("layer_id",layer_id)
        # print("self.cache.size",self.cache_size)
        # print("fanout_cache_refresh",fanout_cache_refresh)
        # for i in range(len(self.cache_size)):
        fanout_cache_sample = self.cache_size-fanout_cache_refresh
        # print(fanout_cache_sample)
        # Sample edges to remove from cache
        removed = cached_graph_structure.sample_neighbors(
            seed_nodes,
            #fanout_cache_refresh,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end remove")
        
        # # Compute the difference and update the cache
        # print("removed")
        # # removed = graph_difference(cached_graph_structure, to_remove)
        # print("end removed")
        
        # Add new edges from the disk to the cache
        # print("add graph")
        to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end add graph")
        
        # # Merge the updated cache with new samples
        # print("begin refresh cache")
        # refreshed_cache = dgl.graph((torch.cat([removed.edges()[0], to_add.edges()[0]]),
        #                              torch.cat([removed.edges()[1], to_add.edges()[1]])),
        #                             num_nodes=self.g.number_of_nodes())
        refreshed_cache = dgl.merge([removed, to_add])
        refreshed_cache = dgl.add_self_loop(refreshed_cache)
        # print("end refresh cache")
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
        self.cycle+=1
        if(self.cycle%self.T==0):
            fanout_cache_retrieval = max(fanout * self.alpha for fanout in self.fanouts)
            fanout_disk = max(self.fanouts) - fanout_cache_retrieval
            fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)

            # print("fanout_size:",fanout_disk)
            self.cached_graph_structure = self.refresh_cache(self.cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
        for i, (fanout) in enumerate(reversed(self.fanouts)):

            # # Refresh cache partially
            # self.cached_graph_structures[i] = self.refresh_cache(i, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            
            # Sample from cache
            frontier_cache = self.cached_graph_structure.sample_neighbors(
                seed_nodes,
                #fanout_cache_retrieval,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            # print("merged_cache",frontier_cache)

            merged_frontier = frontier_cache
            
            # # Sample remaining from disk
            # frontier_disk = g.sample_neighbors(
            #     seed_nodes,
            #     fanout_disk,
            #     edge_dir=self.edge_dir,
            #     prob=self.prob,
            #     replace=self.replace,
            #     output_device=self.output_device,
            #     exclude_edges=self.exclude_eids,
            # )
            # print("frontier_disk",frontier_disk)
            
            # # Merge frontiers
            # merged_frontier = dgl.merge([frontier_cache, frontier_disk]) #merge batch
            # print(f"Merged frontier edges:", merged_frontier.edges())
            
            # Convert the merged frontier to a block
            block = to_block(merged_frontier, seed_nodes)
            if EID in merged_frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = merged_frontier.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

            # blocks = blocks[::-1]
        

        return seed_nodes,output_nodes, blocks


class double_hete(BlockSampler):

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None
        self.cached_struct = None
        self.cycle = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                print("hiiiiiiii is dict")
                for ntype in list(seed_nodes.keys()):
                    print("seed dict",seed_nodes.keys)
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                print("hiiiiiiii is dglgraphobject")
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks
        
        if self.cycle % 20 == 0:
            self.cached_struct=[]
            for i in range(0,len(self.fanouts)):
                frontier1 = g.sample_neighbors(
                    seed_nodes,
                    self.fanouts[i]*2,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=exclude_eids,
                )
                self.cached_struct.append(frontier1)

        k=len(self.fanouts)-1
        for fanout in reversed(self.fanouts):
            print("seeds nodes:",seed_nodes)
            print("org g:",g)
            frontier = self.cached_struct[k].sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            k-=1
            print("sampled frontier:",frontier)
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

class double_shared_cache_hete(BlockSampler):

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None
        self.frontier1 = None
        self.cycle = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                print("hiiiiiiii is dict")
                for ntype in list(seed_nodes.keys()):
                    print("seed dict",seed_nodes.keys)
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                print("hiiiiiiii is dglgraphobject")
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks
        
        if self.cycle % 20 == 0:
            self.frontier1 = g.sample_neighbors(
                seed_nodes,
                fanout*2,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )

        for fanout in reversed(self.fanouts):
            print("seeds nodes:",seed_nodes)
            print("org g:",g)
            frontier = self.frontier1.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            print("sampled frontier:",frontier)
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

class pre_hete(BlockSampler):

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None
        self.frontier1 = None
        self.cycle = 0
    
    def cached(self,g,exclude_eids):
        self.frontier1 = g.sample_neighbors(
                #{'paper':list(range(0,100))},
                {'paper':list(range(0, g.num_nodes("paper")))},
                50,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
        )
        

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                print("hiiiiiiii is dict")
                for ntype in list(seed_nodes.keys()):
                    print("seed dict",seed_nodes.keys)
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                print("hiiiiiiii is dglgraphobject")
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks
        
        # frontier1 = g.sample_neighbors(
        #         #{'paper':list(range(0,100))},
        #         {'paper':list(range(0, g.num_nodes("paper")))},
        #         50,
        #         edge_dir=self.edge_dir,
        #         prob=self.prob,
        #         replace=self.replace,
        #         output_device=self.output_device,
        #         exclude_edges=exclude_eids,
        # )
        if(self.cycle%50 == 0):
            self.cached(g,exclude_eids)
        
        for fanout in reversed(self.fanouts):
            print("seeds nodes:",seed_nodes)
            print("org g:",g)
            frontier = self.frontier1.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            print("sampled frontier:",frontier)
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

class pre_khop_hete(BlockSampler):

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None
        self.cycle = 0
        self.cached_structure = None
    
    def cached(self,g,exclude_eids):
        self.cached_structure=[]
        for i in range(0,len(self.fanouts)):
            frontier1 = g.sample_neighbors(
                    {'paper':list(range(0, g.num_nodes("paper")))},
                    50,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=exclude_eids,
            )
            self.cached_structure.append(frontier1)
        

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                print("hiiiiiiii is dict")
                for ntype in list(seed_nodes.keys()):
                    print("seed dict",seed_nodes.keys)
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                print("hiiiiiiii is dglgraphobject")
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks
        
        if(self.cycle%50 == 0):
            self.cached(g,exclude_eids)
        
        k = len(self.fanouts)-1
        for fanout in reversed(self.fanouts):
            print("seeds nodes:",seed_nodes)
            print("org g:",g)
            fa = int(fanout/2)
            frontier_s1 = self.cached_structure[k].sample_neighbors(
                seed_nodes,
                fa,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            frontier_s2=g.sample_neighbors(
                seed_nodes,
                fanout-fa,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            frontier=dgl.merge([frontier_s1,frontier_s2])
            k-=1
            print("sampled frontier:",frontier)
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

class pre_khop_partial_hete(BlockSampler):

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None
        self.cycle = 0
        self.cached_structure = None
        self.cache_refresh_rate = [0.15 for i in range(len(self.fanouts))]
        self.T = [50 for i in range(len(self.fanouts))]
        self.amplication_rate = [2 for i in range(len(self.fanouts))]
    
    def init_cache(self,g,exclude_eids):
        self.cached_structure=[]
        for i in range(0,len(self.fanouts)):
            frontier1 = g.sample_neighbors(
                    {'paper':list(range(0, g.num_nodes("paper")))},
                    self.fanouts[i]*self.amplication_rate[i],
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=exclude_eids,
            )
            self.cached_structure.append(frontier1)
    
    def refresh_cache(self, g, exclude_eids, fanout, alpha, k, seed_nodes):
        fa = int(fanout*alpha)
        ufa = fanout-fa
        frontier_s1 = self.cached_structure[k].sample_neighbors(
                seed_nodes,
                ufa,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
        frontier_s2=g.sample_neighbors(
                seed_nodes,
                fa,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
        merged_frontier = dgl.merge([frontier_s1,frontier_s2])
        return merged_frontier
    
    def cache_retrieval(self, exclude_eids, fanout, k, seed_nodes):
        frontier_s1 = self.cached_structure[k].sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
        return frontier_s1

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                print("hiiiiiiii is dict")
                for ntype in list(seed_nodes.keys()):
                    print("seed dict",seed_nodes.keys)
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                print("hiiiiiiii is dglgraphobject")
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks
        
        if(self.cycle == 0):
            self.init_cache(g,exclude_eids)
        
        k = len(self.fanouts)-1
        for fanout in reversed(self.fanouts):
            print("seeds nodes:",seed_nodes)
            print("org g:",g)
            if(self.cycle%self.T[k] == 0):
                frontier = self.refresh_cache(g,exclude_eids,fanout,self.cache_refresh_rate[k],k,seed_nodes)
            else:
                frontier = self.cache_retrieval(exclude_eids,fanout,k,seed_nodes)
            k-=1
            print("sampled frontier:",frontier)
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        
        self.cycle+=1

        return seed_nodes, output_nodes, blocks
    

class pre_khop_full_hete(BlockSampler):

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None
        self.cycle = 0
        self.cached_structure = None
        self.cache_refresh_rate = [1 for i in range(len(self.fanouts))]
        self.T = [50 for i in range(len(self.fanouts))]
        self.amplication_rate = [2 for i in range(len(self.fanouts))]
        self.hete_class = "paper"
    
    def init_cache(self,g,exclude_eids):
        self.cached_structure=[]
        for i in range(0,len(self.fanouts)):
            frontier1 = g.sample_neighbors(
                    {self.hete_class:list(range(0, g.num_nodes()))},
                    self.fanouts[i]*self.amplication_rate[i],
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=exclude_eids,
            )
            self.cached_structure.append(frontier1)
    
    def refresh_cache(self, g, exclude_eids, fanout, alpha, k, seed_nodes):
        fa = int(fanout*alpha)
        ufa = fanout-fa
        frontier_s1 = self.cached_structure[k].sample_neighbors(
                seed_nodes,
                fa,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
        frontier_s2=g.sample_neighbors(
                seed_nodes,
                ufa,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
        merged_frontier = dgl.merge([frontier_s1,frontier_s2])
        return merged_frontier
    
    def cache_retrieval(self, exclude_eids, fanout, k, seed_nodes):
        frontier_s1 = self.cached_structure[k].sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
        return frontier_s1

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                print("hiiiiiiii is dict")
                for ntype in list(seed_nodes.keys()):
                    print("seed dict",seed_nodes.keys)
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                print("hiiiiiiii is dglgraphobject")
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks
        
        if(self.cycle == 0):
            self.init_cache(g,exclude_eids)
        
        k = len(self.fanouts)-1
        for fanout in reversed(self.fanouts):
            print("seeds nodes:",seed_nodes)
            print("org g:",g)
            if(self.cycle%self.T[k] == 0):
                frontier = self.refresh_cache(g,exclude_eids,fanout,self.cache_refresh_rate[k],k,seed_nodes)
            else:
                frontier = self.cache_retrieval(exclude_eids,fanout,k,seed_nodes)
            k-=1
            print("sampled frontier:",frontier)
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        
        self.cycle+=1

        return seed_nodes, output_nodes, blocks

class NeighborSampler_OTF_struct_shared_cache_hete_mo(BlockSampler):
    def __init__(self, 
                fanouts, 
                edge_dir='in', 
                alpha=0.6, 
                beta=2, 
                gamma=0.4, 
                T=50,
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
        # self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids

        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.fused = fused
        self.mapping = {}
        self.cache_size = max([fanout * alpha * beta for fanout in fanouts])
        # print(self.cache_size)

        # Initialize cache with amplified fanouts
        self.cached_graph_structure = None #self.initialize_cache(self.cache_size) 
        self.T = T
        self.cycle = 0

    def initialize_cache(self, g, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        # print("begin init")
        cached_graph = g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes(), dtype=torch.int64),
            # torch.arange(0, self.g.number_of_nodes()),
            {'paper':list(range(0, g.num_nodes("paper")))},
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

    def refresh_cache(self,cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        print("begin refresh")
        # print("begin remove")
        fanout_cache_sample=[]
        # print("layer_id",layer_id)
        # print("self.cache.size",self.cache_size)
        # print("fanout_cache_refresh",fanout_cache_refresh)
        # for i in range(len(self.cache_size)):
        fanout_cache_sample = self.cache_size-fanout_cache_refresh
        # print(fanout_cache_sample)
        # Sample edges to remove from cache
        removed = cached_graph_structure.sample_neighbors(
            seed_nodes,
            #fanout_cache_refresh,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end remove")
        
        # # Compute the difference and update the cache
        # print("removed")
        # # removed = graph_difference(cached_graph_structure, to_remove)
        # print("end removed")
        
        # Add new edges from the disk to the cache
        # print("add graph")
        to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # print("end add graph")
        
        # # Merge the updated cache with new samples
        # print("begin refresh cache")
        # refreshed_cache = dgl.graph((torch.cat([removed.edges()[0], to_add.edges()[0]]),
        #                              torch.cat([removed.edges()[1], to_add.edges()[1]])),
        #                             num_nodes=self.g.number_of_nodes())
        refreshed_cache = dgl.merge([removed, to_add])
        # refreshed_cache = dgl.add_self_loop(refreshed_cache)
        print("end refresh cache")
        return refreshed_cache

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        if(self.cycle%self.T==0):
            self.cached_graph_structure = self.initialize_cache(g,self.cache_size)
        # print("in sample blocks")
        self.cycle+=1
        if(self.cycle%self.T==0):
            fanout_cache_retrieval = max(fanout * self.alpha for fanout in self.fanouts)
            fanout_disk = max(self.fanouts) - fanout_cache_retrieval
            fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)

            # print("fanout_size:",fanout_disk)
            self.cached_graph_structure = self.refresh_cache(self.cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
        for i, (fanout) in enumerate(reversed(self.fanouts)):

            # # Refresh cache partially
            # self.cached_graph_structures[i] = self.refresh_cache(i, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            
            # Sample from cache
            frontier_cache = self.cached_graph_structure.sample_neighbors(
                seed_nodes,
                #fanout_cache_retrieval,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            # print("merged_cache",frontier_cache)

            merged_frontier = frontier_cache
            
            # # Sample remaining from disk
            # frontier_disk = g.sample_neighbors(
            #     seed_nodes,
            #     fanout_disk,
            #     edge_dir=self.edge_dir,
            #     prob=self.prob,
            #     replace=self.replace,
            #     output_device=self.output_device,
            #     exclude_edges=self.exclude_eids,
            # )
            # print("frontier_disk",frontier_disk)
            
            # # Merge frontiers
            # merged_frontier = dgl.merge([frontier_cache, frontier_disk]) #merge batch
            # print(f"Merged frontier edges:", merged_frontier.edges())
            
            # Convert the merged frontier to a block
            block = to_block(merged_frontier, seed_nodes)
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        

        return seed_nodes,output_nodes, blocks

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
        self.alpha = alpha # alpha is cache amplication rate, e.g: 2
        self.beta = beta # beta is efficient cache usage rate, e.g: 0.75 (so we can storage smaller cache size)
        self.gamma = gamma # gamma is cache sample rate, e.g: 0.5 (so we can sample (2*0.75*0.5)=0.75 from cache and 0.25 from disk)
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
        self.cache_size = [int(fanout * alpha * beta) for fanout in fanouts]
        # print(self.cache_size)

        # Initialize cache with amplified fanouts
        self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

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

    def refresh_cache(self,layer_id, cached_graph_structure,fanout_cache_retrieval, fanout_cache_refresh,mode="full"):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        if (mode=="partial"):
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
        elif(mode=="full"):
            refreshed_cache = cached_graph_structure.sample_neighbors(
                torch.arange(0, self.g.number_of_nodes()),
                self.cache_size[layer_id],
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
        else:
            raise TypeError("mode must be partial or full")
        refreshed_cache = dgl.add_self_loop(refreshed_cache)
        return refreshed_cache


    def cache_retrieval(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_disk_retrieval):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        # fanout_cache_retrieval = min(fanout_cache_retrieval, self.cache_size[layer_id])
        cache_remain_compute = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_retrieval,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("cache remain",cache_remain_compute)
        disk_to_add_compute = self.g.sample_neighbors(
            seed_nodes,
            fanout_disk_retrieval,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("disk to add",disk_to_add_compute)
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
            # fanout_cache_retrieval = self.cache_size[i]*self.gamma
            # fanout_disk_retrieval = self.cache_size[i] - fanout_cache_retrieval
            # fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)
            if (self.cycle%(self.Toptim*len(self.fanouts))==0):
                # Refresh cache partially/full
                fanout_cache_retrieval = min(int(self.cache_size[i]*self.gamma),self.cache_size[i]-1)
                fanout_disk_retrieval = self.cache_size[i] - fanout_cache_retrieval
                fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)
                print("call the cache refresh")
                self.cached_graph_structures[i]= self.refresh_cache(i, cached_graph_structure, fanout_cache_retrieval, fanout_cache_refresh)
            
            if (self.cycle%self.T==0):
                # sample from cache and disk partially
                print("call the cache disk partial retrieval")
                fanout_cache_retrieval = min(int(self.cache_size[i]*self.gamma),fanout-1)
                fanout_disk_retrieval = fanout - fanout_cache_retrieval

                print("cache retrieval",fanout_cache_retrieval)
                print("disk retrieval",fanout_disk_retrieval)

                frontier_cache = self.cache_retrieval(i, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_disk_retrieval)
                # frontier_cache = self.cached_graph_structures[i].sample_neighbors(
                #     seed_nodes,
                #     #fanout_cache_retrieval,
                #     fanout,
                #     edge_dir=self.edge_dir,
                #     prob=self.prob,
                #     replace=self.replace,
                #     output_device=self.output_device,
                #     exclude_edges=self.exclude_eids,
                # )
            else:
                # Sample from cache fully
                print("call the cache full retrieval")
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
                print("full sample",frontier_cache)

            self.cycle+=1
            
            # Convert the merged frontier to a block
            block = to_block(frontier_cache, seed_nodes)
            if EID in frontier_cache.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_cache.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks