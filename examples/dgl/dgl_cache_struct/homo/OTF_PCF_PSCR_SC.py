# each time partial refresh and partial fetch

class NeighborSampler_OTF_struct_PCFPSCR_SC(BlockSampler):
    """
    Implements an on-the-fly (OTF) neighbor sampling strategy for Deep Graph Library (DGL) graphs. 
    This sampler dynamically samples neighbors while balancing efficiency through caching and 
    freshness of samples by periodically refreshing parts of the cache. It supports specifying 
    fanouts, sampling direction, and probabilities, along with cache management parameters to 
    control the trade-offs between sampling efficiency and cache freshness.

    As for the parameters explanations,
    1. amp_rate: sample a larger cache than the original cache to store the local structure
    2. refresh_rate: decide how many portion should be sampled from disk, and the remaining comes out from cache, then combine them as new disk
    3. T: decide how long time will the cache to refresh and store the new structure (refresh mode in OTF is partially refresh)
    """
    
    def __init__(self, g, 
                fanouts, 
                edge_dir='in', 
                amp_rate=1.5, # cache amplification rate (should be bigger than 1 --> to sample for multiple time)
                refresh_rate=0.4, #propotion of cache to be refresh, should be a positive float smaller than 0.5
                T=50, # refresh time, for example
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
        self.amp_rate = amp_rate
        self.refresh_rate = refresh_rate
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
        self.amp_cache_size = [fanout * amp_rate for fanout in fanouts]
        self.T = T
        self.cycle = 0
        # self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

        self.shared_cache_size = max(self.amp_cache_size)
        self.shared_cache = self.initialize_cache(self.shared_cache_size)

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            torch.arange(0, self.g.number_of_nodes()),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_rf_cache(self,layer_id, seed_nodes, fanout_cache_refresh, fanout):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_remain = self.shared_cache_size-fanout_cache_refresh
        fanout_cache_pr = fanout-fanout_cache_refresh

        all_nodes = torch.arange(0, self.g.number_of_nodes())
        # mask = ~torch.isin(all_nodes, seed_nodes)
        # # 使用布尔掩码来选择不在seed_nodes中的节点
        # unchanged_nodes = all_nodes[mask]

        # unchanged_nodes = torch.arange(0, self.g.number_of_nodes())-seed_nodes
        # the rest node structure remain the same
        unchanged_structure = self.shared_cache.sample_neighbors(
            all_nodes,
            # unchanged_nodes,
            self.shared_cache_size,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        # the OTF node structure should 
        changed_cache_remain = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_remain,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        cache_pr = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_pr,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        changed_disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        refreshed_cache = dgl.merge([unchanged_structure, changed_cache_remain, changed_disk_to_add])
        retrieval_cache = dgl.merge([cache_pr, changed_disk_to_add])
        return refreshed_cache, retrieval_cache

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes
        self.cycle += 1
        for i, (fanout) in enumerate(reversed(self.fanouts)):
            fanout_cache_refresh = int(fanout * self.refresh_rate)

            # Refresh cache&disk partially, while retrieval cache&disk partially
            if(self.cycle%self.T==0):
                self.shared_cache, frontier_comp = self.OTF_rf_cache(i, seed_nodes, fanout_cache_refresh, fanout)
            else:
                frontier_comp = self.shared_cache.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            
            # Convert the merged frontier to a block
            block = to_block(frontier_comp, seed_nodes)
            if EID in frontier_comp.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_comp.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        return seed_nodes,output_nodes, blocks