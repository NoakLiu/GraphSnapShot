class NeighborSampler_OTF_fetch_struct_shared_cache_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            amp_rate=2, 
            fetch_rate = 0.4,
            T_refresh=None,
            refresh_rate=0.4,
            T_fetch=3, # fetch period of time
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

        self.alpha = amp_rate
        self.cycle = 0  # Initialize sampling cycle counter
        self.sc_size = max([f * amp_rate for f in fanouts])  # Amplified fanouts for pre-sampling
        self.refresh_rate = refresh_rate
        if T_refresh!=None:
            self.T_refresh = T_refresh
        else:
            self.T_refresh = int(self.g.number_of_nodes()/max(self.fanouts) *self.amp_rate)
        self.T_fetch = T_fetch
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.fetch_rate = fetch_rate
        # self.cache_struct = []  # Initialize cache structure
        self.hete_label = hete_label
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache
        self.shared_cache = self.full_cache_refresh(self.sc_size)
    
    def full_cache_refresh(self, fanout_cache_storage, exclude_eids = None):
        cached_graph = self.g.sample_neighbors(
            # torch.arange(0, self.g.number_of_nodes()),
            {self.hete_label:list(range(0, self.g.num_nodes(self.hete_label)))},
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
        )
        print("cache refresh")
        return cached_graph

    def OTF_fetch(self,layer_id,  seed_nodes, fanout_cache_fetch, exclude_eids = None):
        print("OTF fetch cache")
        if(fanout_cache_fetch==self.fanouts[layer_id]):
            cache_fetch = self.shared_cache.sample_neighbors(
            seed_nodes,
            fanout_cache_fetch,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=exclude_eids,
            )
            return cache_fetch
        else:
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
            return OTF_fetch_res

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
        print("self.T_refresh=",self.T_refresh)
        # refresh full cache after a period of time
        if((self.cycle%self.T_refresh)==0):
            self.shared_cache = self.full_cache_refresh(self.sc_size)

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
                    block = self.g.sample_neighbors_fused(
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
            # cached_structure = self.shared_cache
            fanout = self.fanouts[k]

            fanout_cache_fetch = int(fanout * self.fetch_rate)

            # fetch cache partially
            if((self.cycle%self.T_fetch)==0):
                frontier_OTF = self.OTF_fetch(k, seed_nodes, fanout_cache_fetch)
            else:
                #frontier_OTF = self.OTF_fetch(i, seed_nodes, self.fanouts[i])
                frontier_OTF = self.shared_cache.sample_neighbors(
                    seed_nodes,
                    fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=self.exclude_eids,
                )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier_OTF, seed_nodes)
            if EID in frontier_OTF.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier_OTF.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks