import dgl
from dgl.dataloading import NeighborSampler
import torch

class NeighborSampler_OTF(NeighborSampler):
    def __init__(self, g, alpha, beta, k, fanouts, **kwargs):
        """
        Initialize the on-the-fly neighbor sampler.

        Parameters:
        g (DGLGraph): The input graph.
        alpha (float): Fraction of nodes to include in the static subgraph (g_static).
        beta (float): Fraction of g_static nodes to be refreshed every k epochs.
        k (int): Interval (in epochs) at which the static subgraph is partially refreshed.
        fanouts (list[int] or list[dict[etype, int]]): The number of neighbors to sample for each layer.
        **kwargs: Additional keyword arguments for the NeighborSampler.
        """
        super().__init__(fanouts, **kwargs)
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.epoch_counter = 0

        # Preprocess to split the graph into static and dynamic parts
        self.preprocess()

    def preprocess(self):
        """Split the input graph into static (g_static) and dynamic (g_dynamic) subgraphs."""
        num_nodes = self.g.number_of_nodes()
        num_static_nodes = int(num_nodes * self.alpha)

        # Randomly sample nodes for the static graph
        static_nodes = torch.randperm(num_nodes)[:num_static_nodes]

        # Create static and dynamic subgraphs
        self.g_static = self.g.subgraph(static_nodes).to('cuda')
        dynamic_nodes = torch.tensor(list(set(range(num_nodes)) - set(static_nodes.tolist())))
        self.g_dynamic = self.g.subgraph(dynamic_nodes)

    def cache_refresh(self):
        """Refresh a fraction (beta) of the static subgraph's nodes by swapping them with nodes from the dynamic subgraph."""
        if self.epoch_counter % self.k == 0:
            num_static_nodes = self.g_static.number_of_nodes()
            num_nodes_to_replace = int(num_static_nodes * self.beta)

            # Nodes to discard from the static graph and to add from the dynamic graph
            nodes_to_discard = torch.randperm(num_static_nodes)[:num_nodes_to_replace]
            nodes_to_add = torch.randperm(self.g_dynamic.number_of_nodes())[:num_nodes_to_replace]

            # Update the static and dynamic subgraphs
            remaining_static_nodes = torch.tensor(list(set(range(num_static_nodes)) - set(nodes_to_discard.tolist())))
            new_static_nodes = torch.cat([self.g_static.ndata[dgl.NID][remaining_static_nodes], self.g_dynamic.ndata[dgl.NID][nodes_to_add]])
            self.g_static = self.g.subgraph(new_static_nodes).to('cuda')

            remaining_dynamic_nodes = torch.tensor(list(set(range(self.g_dynamic.number_of_nodes())) - set(nodes_to_add.tolist())))
            new_dynamic_nodes = torch.cat([self.g_dynamic.ndata[dgl.NID][remaining_dynamic_nodes], self.g_static.ndata[dgl.NID][nodes_to_discard]])
            self.g_dynamic = self.g.subgraph(new_dynamic_nodes)

    def sample_blocks(self, seed_nodes, exclude_eids=None):
        """Sample blocks from both static and dynamic subgraphs and combine the results."""
        # Refresh the cache if needed
        self.cache_refresh()

        # Increment the epoch counter
        self.epoch_counter += 1

        # Sample from the static subgraph
        seed_nodes_static, output_nodes_static, blocks_static = super().sample_blocks(self.g_static, seed_nodes, exclude_eids)
        # Sample from the dynamic subgraph
        seed_nodes_dynamic, output_nodes_dynamic, blocks_dynamic = super().sample_blocks(self.g_dynamic, seed_nodes, exclude_eids)

        # Combine the sampled results from the static and dynamic subgraphs
        combined_seed_nodes, combined_output_nodes, combined_blocks = self._combine_blocks(
            (seed_nodes_static, output_nodes_static, blocks_static),
            (seed_nodes_dynamic, output_nodes_dynamic, blocks_dynamic)
        )

        return combined_seed_nodes, combined_output_nodes, combined_blocks

    def _combine_blocks(self, blocks_static, blocks_dynamic):
        """
        Combine the sampled blocks from static and dynamic subgraphs.

        Returns:
        combined_seed_nodes (list): List containing seed nodes from static and dynamic samplers.
        combined_output_nodes (list): List containing output nodes from static and dynamic samplers.
        combined_blocks (list): List containing blocks from static and dynamic samplers.
        """
        combined_seed_nodes = [blocks_static[0], blocks_dynamic[0]]
        combined_output_nodes = [blocks_static[1], blocks_dynamic[1]]
        combined_blocks = [blocks_static[2], blocks_dynamic[2]]

        return combined_seed_nodes, combined_output_nodes, combined_blocks
