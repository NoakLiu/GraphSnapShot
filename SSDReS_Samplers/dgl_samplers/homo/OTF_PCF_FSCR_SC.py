# each time retrieval the result from disk and cache partially
# after a period of time, refresh the full cache 

class NeighborSampler_OTF_struct_PCFFSCR(BlockSampler):
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
                T_fetch=3, # fetch period of time
                T_refresh=None, # refresh time
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
        if T_refresh!=None:
            self.T_refresh = T_refresh
        else:
            self.T_refresh = int(max([self.cache_size[i]/self.fanouts[i] for i in len(self.fanouts)]))
        self.T_fetch = T_fetch
        # self.cached_graph_structures = None
        self.cycle = 0

        self.shared_cache_size = max(self.amp_cache_size)
        self.shared_cache = None

    def full_cache_refresh(self, fanout_cache_storage):
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
        print("cache refresh")
        return cached_graph

    def OTF_fetch(self,layer_id,  seed_nodes, fanout_cache_fetch):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_disk_fetch = self.fanouts[layer_id]-fanout_cache_fetch
        cache_fetch = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_fetch,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        disk_fetch = self.g.sample_neighbors(
            seed_nodes,
            fanout_disk_fetch,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )

        OTF_fetch_res = dgl.merge([cache_fetch, disk_fetch])
        print("OTF fetch cache")
        return OTF_fetch_res

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Samples blocks for GNN layers by combining cached samples with dynamically sampled 
        neighbors. This method also partially refreshes the cache based on specified parameters 
        to balance between sampling efficiency and the freshness of the samples.
        """
        blocks = []
        output_nodes = seed_nodes

        # refresh full cache after a period of time
        if((self.cycle%self.T_refresh)==0):
            self.shared_cache = self.full_cache_refresh(self.shared_cache_size)
            # self.cached_graph_structures = [self.full_cache_refresh(cache_size) for cache_size in self.cache_size]
        
        for i, (fanout) in enumerate(reversed(self.fanouts)):
            fanout_cache_refresh = int(fanout * self.refresh_rate)

            # fetch cache partially
            if((self.cycle%self.T_fetch)==0):
                frontier_OTF = self.OTF_fetch(i, seed_nodes, fanout_cache_refresh)
            else:
                frontier_OTF = self.OTF_fetch(i, seed_nodes, self.fanouts[i])
            
            # Convert the merged frontier to a block
            block = to_block(frontier_OTF, seed_nodes)
            if EID in frontier_OTF.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_OTF.edata[EID]
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer
        
        self.cycle += 1

        return seed_nodes,output_nodes, blocks