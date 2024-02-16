import dgl
from dgl.dataloading import NeighborSampler
import torch

class NeighborSampler_FCR(NeighborSampler):
    def __init__(self, g, alpha, beta, k, fanouts, **kwargs):
        """
        Initialize the NeighborSampler_FCR.

        Parameters:
        g (DGLGraph): The input graph.
        alpha (float): The fraction of nodes to include in g_static.
        beta (float): The fraction of g_static nodes to replace every k epochs.
        k (int): The number of epochs after which g_static is partially updated.
        fanouts (list[int] or list[dict[etype, int]]): List of neighbors to sample per edge type for each GNN layer.
        **kwargs: Additional arguments for the NeighborSampler.
        """
        super().__init__(fanouts, **kwargs)
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.epoch_counter = 0

        # Preprocess to split the graph
        self.preprocess()

    def preprocess(self):
        """
        Splits the input graph into g_static and g_dynamic based on alpha.
        """
        num_nodes = self.g.number_of_nodes()
        num_static_nodes = int(num_nodes * self.alpha)

        # Randomly sample nodes for g_static
        static_nodes = torch.randperm(num_nodes)[:num_static_nodes]

        # Create g_static and g_dynamic
        g_static = self.g.subgraph(static_nodes).to('cuda')
        dynamic_nodes = torch.tensor(list(set(range(num_nodes)) - set(static_nodes.tolist())))
        g_dynamic = self.g.subgraph(dynamic_nodes)

        self.g_static, self.g_dynamic = g_static, g_dynamic

    def cache_refresh(self):
        """
        Updates g_static by discarding beta fraction of its nodes and replacing
        them with the same number of nodes from g_dynamic through disk_cache_swap.
        """
        num_static_nodes = self.g_static.number_of_nodes()
        num_nodes_to_replace = int(num_static_nodes * self.beta)

        # Nodes to discard from g_static and to add from g_dynamic
        nodes_to_discard = torch.randperm(num_static_nodes)[:num_nodes_to_replace]
        nodes_to_add = torch.randperm(self.g_dynamic.number_of_nodes())[:num_nodes_to_replace]

        # Perform the swap
        self.disk_cache_swap(nodes_to_discard, nodes_to_add)

    def disk_cache_swap(self, nodes_to_discard, nodes_to_add):
        """
        Performs the actual swapping of nodes between g_static and g_dynamic.
        """
        # Update g_static
        remaining_static_nodes = torch.tensor(list(set(range(self.g_static.number_of_nodes())) - set(nodes_to_discard.tolist())))
        new_static_nodes = torch.cat([self.g_static.ndata[dgl.NID][remaining_static_nodes], self.g_dynamic.ndata[dgl.NID][nodes_to_add]])
        self.g_static = self.g.subgraph(new_static_nodes).to('cuda')

        # Update g_dynamic
        remaining_dynamic_nodes = torch.tensor(list(set(range(self.g_dynamic.number_of_nodes())) - set(nodes_to_add.tolist())))
        new_dynamic_nodes = torch.cat([self.g_dynamic.ndata[dgl.NID][remaining_dynamic_nodes], self.g_static.ndata[dgl.NID][nodes_to_discard]])
        self.g_dynamic = self.g.subgraph(new_dynamic_nodes)

    def sample_blocks(self, seed_nodes, exclude_eids=None):
        """
        Overrides the sample_blocks method to incorporate the logic of updating g_static
        every k epochs.
        """
        if self.epoch_counter % self.k == 0:
            self.cache_refresh()

        self.epoch_counter += 1

        # Call the original sample_blocks method using g_static
        return super().sample_blocks(self.g_static, seed_nodes, exclude_eids)
