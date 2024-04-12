class NeighborSampler_OTF_struct_PCFPSCR_shared_cache_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            amp_rate=2, 
            T=20,
            refresh_rate=0.4,
            hete_label=None,
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
        self.g = g
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.exclude_eids = None

        self.amp_rate = amp_rate
        self.hete_label = hete_label
        self.cycle = 0  # Initialize sampling cycle counter
        self.sc_size = max([f * self.amp_rate for f in fanouts])  # Amplified fanouts for pre-sampling
        self.T = T
        self.refresh_rate = refresh_rate
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.shared_cache = self.initialize_cache(fanout_cache_storage=self.sc_size)  # Initialize cache structure
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer with an amplified fanout to pre-sample a larger
        set of neighbors. This pre-sampling helps in reducing the need for dynamic sampling 
        at every iteration, thereby improving efficiency.
        """
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        print("end init cache")
        return cached_graph

    def OTF_rf_cache(self,seed_nodes, fanout_cache_refresh, fanout):
        """
        Refreshes a portion of the cache based on the gamma parameter by replacing some of the 
        cached edges with new samples from the graph. This method ensures the cache remains 
        relatively fresh and reflects changes in the dynamic graph structure or sampling needs.
        """
        fanout_cache_remain = self.sc_size-fanout_cache_refresh
        fanout_cache_pr = fanout-fanout_cache_refresh
        # unchanged_nodes = range(torch.arange(0, self.g.number_of_nodes()))-seed_nodes
        # the rest node structure remain the same
        all_nodes = torch.arange(0,  self.g.num_nodes(self.hete_label))
        print("seed nodes:",seed_nodes)
        print("all nodes",all_nodes)
        mask = ~torch.isin(all_nodes, seed_nodes[self.hete_label])
        # bool mask to select those nodes do not in seed_nodes
        unchanged_nodes = {self.hete_label: all_nodes[mask]}
        unchanged_structure = self.shared_cache.sample_neighbors(
            unchanged_nodes,
            self.sc_size,
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

        # # refresh cache after a period of time for generalization
        # self.Toptim = int(g.number_of_nodes() / max(self.amplified_fanouts))
        # if self.cycle % self.Toptim == 0:
        #     self.cache_refresh(g)  # Refresh cache every T cycles
        
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

        for k in range(len(self.fanouts)-1,-1,-1):
            fanout_cache_refresh = int(self.fanouts[k] * self.refresh_rate)

            # Refresh cache&disk partially, while retrieval cache&disk partially
            self.shared_cache, frontier_comp = self.OTF_rf_cache( seed_nodes, fanout_cache_refresh, self.fanouts[k])

            # Sample frontier from the cache for acceleration
            block = to_block(frontier_comp, seed_nodes)
            if EID in frontier_comp.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_comp.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks