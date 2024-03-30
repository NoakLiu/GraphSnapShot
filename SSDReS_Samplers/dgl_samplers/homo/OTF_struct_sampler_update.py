class NeighborSampler_OTF_struct_update(BlockSampler):
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
                T=50, # refresh time
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
        self.cache_size = [fanout * amp_rate for fanout in fanouts]
        self.T = T
        self.cached_graph_structures = [self.initialize_cache(cache_size) for cache_size in self.cache_size]

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
            mappings=self.mapping
        )
        print("end init cache")
        return cached_graph

    def refresh_cache(self,layer_id, cached_graph_structure, seed_nodes, fanout_cache_refresh):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_sample = self.cache_size[layer_id]-fanout_cache_refresh
        cache_remain = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_sample,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        disk_to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        refreshed_cache = dgl.merge([cache_remain, disk_to_add])
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
        for i, (fanout, cached_graph_structure) in enumerate(zip(reversed(self.fanouts), reversed(self.cached_graph_structures))):
            fanout_cache_refresh = int(fanout * self.refresh_rate)

            # Refresh cache partially
            self.cached_graph_structures[i] = self.refresh_cache(i, cached_graph_structure, seed_nodes, fanout_cache_refresh)
            
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