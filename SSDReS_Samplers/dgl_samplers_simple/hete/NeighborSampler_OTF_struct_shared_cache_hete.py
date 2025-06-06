from .. import backend as F
from ..base import EID, NID
from ..heterograph import DGLGraph
from ..transforms import to_block
from ..utils import get_num_threads
from .base import BlockSampler
import torch
import dgl

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

class NeighborSampler_OTF_struct_shared_cache_hete_mo(BlockSampler):
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
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        

        return seed_nodes,output_nodes, blocks
