import torch
import dgl

# homogeneous graph
g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))
g_edges = g.all_edges(form='all')
sg = dgl.sampling.sample_neighbors(g, [0, 1], 3, exclude_edges=[0, 1, 2])
sg.all_edges(form='all')
print(sg.has_edges_between(g_edges[0][:3],g_edges[1][:3]))


# heteogeneous graph
g = dgl.heterograph({
    ('drug', 'interacts', 'drug'): ([0, 0, 1, 1, 3, 2], [1, 2, 0, 1, 2, 0]),
    ('drug', 'interacts', 'gene'): ([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]),
    ('drug', 'treats', 'disease'): ([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0])})
g_edges = g.all_edges(form='all', etype=('drug', 'interacts', 'drug'))
print(g_edges) # tuple of 3 tensors: src node tensor, dst node tensor, edges id tensor
excluded_edges  = {('drug', 'interacts', 'drug'): g_edges[2][:3]}
print(excluded_edges) # select specific edges id, here discard previous 3 edges
sg = dgl.sampling.sample_neighbors(g, {'drug':[0, 1]}, 3, exclude_edges=excluded_edges, copy_ndata=True, copy_edata=True) # seed nodes only consider the drug type node with specific node IDs
sgg = sg.sample_neighbors({'drug':[1]}, 3, exclude_edges=excluded_edges)
print(sg.all_edges(form='all', etype=('drug', 'interacts', 'drug')))
print(sg.has_edges_between(g_edges[0][:3],g_edges[1][:3],etype=('drug', 'interacts', 'drug')))


