from .. import backend as F
from ..base import EID, NID
from ..heterograph import DGLGraph
from ..transforms import to_block
from ..utils import get_num_threads
from .base import BlockSampler
import torch
import dgl

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
            print("large")
            print(fanout)
            print("---")
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
            print(frontier)
            print(self.cache_struct)
            print("then append")
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
        print("cache struct")
        print(self.cache_struct)
        print(len(self.cache_struct))
        print("---")
        for k in range(len(self.cache_struct)-1,-1,-1):
            frontier_large = self.cache_struct[k]
            fanout = self.fanouts[k]
            print("small")
            print("seed",seed_nodes)
            print("fanout",fanout)
            print("frontier_large",frontier_large)
            print("---")
            frontier = frontier_large.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # Directly use pre-sampled frontier from the cache
            block = to_block(frontier, seed_nodes)
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks
