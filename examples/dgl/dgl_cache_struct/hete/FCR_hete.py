class NeighborSampler_FCR_struct_hete(BlockSampler):    
    def __init__(
            self, 
            g,
            fanouts, 
            edge_dir='in', 
            alpha=2, 
            T=20,
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

        self.alpha = alpha
        self.cycle = 0  # Initialize sampling cycle counter
        self.amplified_fanouts = [f * alpha for f in fanouts]  # Amplified fanouts for pre-sampling
        self.T = T
        self.Toptim = None # int(self.g.number_of_nodes() / max(self.amplified_fanouts))
        self.cache_struct = []  # Initialize cache structure
        self.hete_label = hete_label
        # self.cache_refresh(self.g)  # Pre-sample and populate the cache

    def cache_refresh(self,g,exclude_eids=None):
        """
        Pre-samples neighborhoods with amplified fanouts and refreshes the cache. This method
        is automatically called upon initialization and after every T sampling iterations to
        ensure that the cache is periodically updated with fresh samples.
        """
        self.cache_struct.clear()  # Clear existing cache
        for fanout in self.amplified_fanouts:
            # Sample neighbors for each layer with amplified fanout
            frontier = g.sample_neighbors(
                {self.hete_label:list(range(0, g.num_nodes(self.hete_label)))},
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )
            # frontier = dgl.add_self_loop(frontier)
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

        # refresh cache after a period of time for generalization
        self.Toptim = int(g.number_of_nodes() / max(self.amplified_fanouts))
        if self.cycle % self.Toptim == 0:
            self.cache_refresh(g)  # Refresh cache every T cycles
        
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

        for k in range(len(self.cache_struct)-1,-1,-1):
            cached_structure = self.cache_struct[k]
            fanout = self.fanouts[k]
            frontier = cached_structure.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids
            )

            # Sample frontier from the cache for acceleration
            block = to_block(frontier, seed_nodes)
            if EID in frontier.edata.keys():
                print("--------in this EID code---------")
                block.edata[EID] = frontier.edata[EID]
            blocks.insert(0, block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

        # output_nodes = seed_nodes
        return seed_nodes, output_nodes, blocks