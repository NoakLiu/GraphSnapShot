import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from scipy.sparse import csgraph
import scipy.sparse.linalg
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
from collections import defaultdict

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(mx):#D^(-1/2)^T*A*D^(-1/2) preprocess 得到一个领域传播位权矩阵
    """
    Row-normalize sparse matrix
    """
    """
    normalize_adj process
    adj--original matrix shape (2708, 2708)
    adj--degree sum matrix shape (2708, 1)
    adj--degree inv matrix shape (2708,)
    adj--degree no inf matrix shape (2708,)
    adj--degree norm diag matrix shape (2708, 2708)
    adj--res shape (2708, 2708)
    """
    ## print("normalize_adj process")
    ## print("adj--original matrix shape",mx.shape)
    rowsum = np.array(mx.sum(1)) #每行求和，求得每个节点的出度列矩阵
    ## print("adj--degree sum matrix shape",rowsum.shape)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    ## print("adj--degree inv matrix shape",r_inv_sqrt.shape)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    ## print("adj--degree no inf matrix shape",r_inv_sqrt.shape)
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    ## print("adj--degree norm diag matrix shape",r_mat_inv_sqrt.shape)
    resm = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()
    ## print("adj--res shape",resm.shape)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo() #D^(-1/2)*A*D^(-1/2)^T

def normalize(mx):#D^-(1)*A
    """Row-normalize sparse matrix"""
    """
    normalize process
    mx--original matrix shape (2708, 1433)
    mx--degree sum matrix shape (2708, 1)
    mx--degree inv matrix shape (2708,)
    mx--degree no inf matrix shape (2708,)
    mx--degree norm diag matrix shape (2708, 2708)
    mx--res shape (2708, 1433)
    """  
    ## print("normalize process")
    ## print("mx--original matrix shape",mx.shape)
    rowsum = np.array(mx.sum(1))
    ## print("mx--degree sum matrix shape",rowsum.shape)
    r_inv = np.power(rowsum, -1).flatten()
    ## print("mx--degree inv matrix shape",r_inv.shape)
    r_inv[np.isinf(r_inv)] = 0.
    ## print("mx--degree no inf matrix shape",r_inv.shape)
    r_mat_inv = sp.diags(r_inv)
    ## print("mx--degree norm diag matrix shape",r_mat_inv.shape)
    mx = r_mat_inv.dot(mx)
    ## print("mx--res shape",mx.shape)

    print("NA.shape",mx.shape)

    print("NA.dtype",mx.dtype)

    print("NA.type",type(mx))
    return mx

def laplacian(mx, norm):
    """Laplacian-normalize sparse matrix"""
    assert (all (len(row) == len(mx) for row in mx)), "Input should be a square matrix"

    return csgraph.laplacian(mx, normed = norm)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data(path="Data", dataset="cora",mode="matrix"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        #Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    graph_dict = nx.from_dict_of_lists(graph)
    print("below is graph_dict")
    # print(graph_dict.shape)
    print(graph_dict)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum()/2))

    features = normalize(features) #将feature按照度归一化
    adj = normalize_adj(adj + sp.eye(adj.shape[0])) #将A按照两个边端点度归一化 PPR方法
    
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)
    adj = torch.FloatTensor(np.array(adj.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end+1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)
    if(mode=="dict"):
        adj_defaultdict = defaultdict(set, {k: set(v) for k, v in graph.items()})
        return adj_defaultdict, features, labels, idx_train, idx_val, idx_test

    return adj, features, labels, idx_train, idx_val, idx_test