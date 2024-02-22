import dgl
import torch
from dgl.sampling import sample_neighbors
from dgl.transforms import to_block

def graph_difference(g1, g2):
    """
    Compute the difference between two graphs, returning a new graph
    that contains only the edges in g1 not present in g2.
    """
    # Convert edges to sets of tuples for easy comparison
    edges_g1 = set(zip(*g1.edges()))
    edges_g2 = set(zip(*g2.edges()))
    
    # Find edges present in g1 but not in g2
    unique_edges = edges_g1 - edges_g2
    unique_src, unique_dst = zip(*unique_edges) if unique_edges else ([], [])
    
    # Create a new graph from the unique edges
    g_unique = dgl.graph((torch.tensor(unique_src), torch.tensor(unique_dst)), num_nodes=g1.number_of_nodes())
    return g_unique

class NeighborSampler_OTF_struct:
    """
    A class that performs on-the-fly neighbor sampling from a DGL graph, utilizing a hybrid approach
    that combines cached sampling and dynamic sampling. It supports partial cache refreshing to maintain
    a balance between sampling efficiency and the freshness of the sampled neighborhoods.
    """
    
    def __init__(self, g, fanouts, edge_dir='in', alpha=0.6, beta=2, gamma=0.4, prob=None, replace=False, output_device=None, exclude_eids=None):
        self.g = g
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prob = prob
        self.replace = replace
        self.output_device = output_device
        self.exclude_eids = exclude_eids
        
        # Initialize cache with amplified fanouts
        self.cached_graph_structures = [self.initialize_cache(fanout * alpha * beta) for fanout in fanouts]

    def initialize_cache(self, fanout_cache_storage):
        """
        Initializes the cache for each layer based on the amplified fanouts.
        """
        cached_graph = sample_neighbors(
            self.g,
            torch.arange(0, self.g.number_of_nodes(), dtype=torch.int64),
            fanout_cache_storage,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        return cached_graph

    def refresh_cache(self, cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh):
        """
        Refreshes the cache by removing a portion of cached edges and adding new edges
        from the disk (original graph) to the cache.
        """
        # Sample edges to remove from cache
        to_remove = cached_graph_structure.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        
        # Compute the difference and update the cache
        removed = graph_difference(cached_graph_structure, to_remove)
        
        # Add new edges from the disk to the cache
        to_add = self.g.sample_neighbors(
            seed_nodes,
            fanout_cache_refresh,
            edge_dir=self.edge_dir,
            prob=self.prob,
            replace=self.replace,
            output_device=self.output_device,
            exclude_edges=self.exclude_eids,
        )
        
        # Merge the updated cache with new samples
        refreshed_cache = dgl.graph((torch.cat([removed.edges()[0], to_add.edges()[0]]),
                                     torch.cat([removed.edges()[1], to_add.edges()[1]])),
                                    num_nodes=self.g.number_of_nodes())
        return refreshed_cache

    def sample_blocks(self, seed_nodes):
        """
        Performs neighbor sampling, combining cached sampling with dynamic sampling,
        and refreshes the cache partially as defined by gamma.
        """
        blocks = []
        for i, (fanout, cached_graph_structure) in enumerate(zip(self.fanouts, self.cached_graph_structures)):
            fanout_cache_retrieval = int(fanout * self.alpha)
            fanout_disk = fanout - fanout_cache_retrieval
            fanout_cache_refresh = int(fanout_cache_retrieval * self.beta * self.gamma)
            
            # Refresh cache partially
            self.cached_graph_structures[i] = self.refresh_cache(cached_graph_structure, seed_nodes, fanout_cache_retrieval, fanout_cache_refresh)
            
            # Sample from cache
            frontier_cache = self.cached_graph_structures[i].sample_neighbors(
                seed_nodes,
                fanout_cache_retrieval,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            
            # Sample remaining from disk
            frontier_disk = self.g.sample_neighbors(
                seed_nodes,
                fanout_disk,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=self.exclude_eids,
            )
            
            # Merge frontiers
            merged_frontier = dgl.batch([frontier_cache, frontier_disk])
            
            # Convert the merged frontier to a block
            block = to_block(self.g, merged_frontier, seed_nodes)
            blocks.append(block)
            seed_nodes = block.srcdata[NID]  # Update seed nodes for the next layer

            print(f"Layer {i}: Merged frontier edges:", merged_frontier.edges())

        return seed_nodes, blocks
