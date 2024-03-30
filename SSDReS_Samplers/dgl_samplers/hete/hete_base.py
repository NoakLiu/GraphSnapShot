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