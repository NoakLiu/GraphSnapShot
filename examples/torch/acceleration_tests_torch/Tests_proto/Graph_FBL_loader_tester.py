
import torch
import torch.optim as optim
from data_processing.data_preprocessing import load_data
import torch.nn.functional as F
from data_processing.data_preprocessing import accuracy
from SSDReS_Samplers.GraphSDSampler import MLP
# from SSDReS_Samplers.GraphKSDSampler_sswp_struct_mask import GraphKSDSampler
from baseline_load.Graph_FBL_load import GraphFBLLoader
from tasks_downstream.KHop_GCN import kHopGCN
from tasks_downstream.KHop_SGC import kHopSGC
import numpy as np
import time
from data_processing.plot import draw_pic

class Trainer:
    def __init__(self, machine, dataset):
        self.machine = machine
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data(dataset=dataset)
        if(self.machine == "gpu"):
            # self.model = MLP(self.features.size(1), max(self.labels) + 1).cuda()
            if(task=="KHhopGCN"):
                self.model = kHopGCN(nfeat=self.features.size(1),
                        nhid=16,
                        nclass=max(self.labels) + 1,
                        dropout=0.4,
                        model_mode="sparse"
                        ).cuda()
            elif(task=="KHopSGC"):
                self.model = kHopSGC(in_features=self.features.size(1),
                        out_features=max(self.labels) + 1,
                        k=3).cuda()
        else:
            # self.model = MLP(self.features.size(1), max(self.labels) + 1)
            if(task=="KHhopGCN"):
                self.model = kHopGCN(nfeat=self.features.size(1),
                        nhid=16,
                        nclass=max(self.labels) + 1,
                        dropout=0.4,
                        model_mode="sparse"
                        )
            elif (task == "KHopSGC"):
                self.model = kHopSGC(in_features=self.features.size(1),
                                     out_features=max(self.labels) + 1,
                                     k=3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self):
        # Training
        # sampler = GraphKSDSampler(list(range(self.adj[self.idx_train].size(0))), self.adj[self.idx_train][:, self.idx_train], presampled_nodes, sampled_depth, presampled_perexpansion)

        for epoch in range(100):
            self.model.train()
            self.optimizer.zero_grad()

            # sampler = GraphKSDSampler(list(range(self.adj.size(0))),
            #                           self.adj, sampled_depth, batch_size=batch_size,
            #                           n=presample_disk, rcn=retrieve_cache, rdn=retrieve_disk, ecn=erase_cache)

            sampler = GraphFBLLoader(list(range(self.adj.size(0))),
                                      self.adj, sampled_depth, batch_size=batch_size,drn=disk_read)


            for p in range(5):
                # if this is because the data loading module
                # Sampling nodes

                # 改进training的时候如何调用各个程序，是否正确调用
                # comp_nodes, comp_list = sampler.resample(resampled_nodes, alpha, 2)
                comp_nodes, comp_list = sampler.FBL_loader()

                #---->这里每次都刷新最初的batch_size进行计算
                # 1. batch进行计算
                # 2. 进行k-hop MLP的计算

                # 这里使用一个k-hop MLP进行实验
                # comp_adj = comp_list[0]#[comp_nodes]#[:, comp_nodes]
                # comp_adj = torch.tensor(comp_adj, dtype=torch.float32)
                comp_features = self.features[comp_nodes]
                comp_labels = self.labels[comp_nodes]

                # Forward
                if(self.machine=="gpu"):
                    output = self.model(comp_features.cuda(), comp_list.cuda())
                    loss = F.nll_loss(output, comp_labels.cuda())
                else:
                    output = self.model(comp_features, comp_list)
                    loss = F.nll_loss(output, comp_labels)

                # Backward
                loss.backward()
                self.optimizer.step()

                # Validation
                self.model.eval()
                with torch.no_grad():
                    if(self.machine=="gpu"):
                        output = self.model(self.features[self.idx_val].cuda(), self.adj[self.idx_val][:, self.idx_val].cuda())
                        val_loss = F.nll_loss(output, self.labels[self.idx_val].cuda())
                        val_acc = accuracy(output, self.labels[self.idx_val].cuda())
                    else:
                        output = self.model(self.features[self.idx_val],
                                            self.adj[self.idx_val][:, self.idx_val])
                        val_loss = F.nll_loss(output, self.labels[self.idx_val])
                        val_acc = accuracy(output, self.labels[self.idx_val])
                    print('Epoch: {:04d}'.format(epoch + 1),'at p: {:04d}'.format(p), 'loss_train: {:.4f}'.format(loss.item()),
                          'loss_val: {:.4f}'.format(val_loss.item()), 'val_acc: {:.4f}'.format(val_acc.item()))

    def test(self):
        # Testing the model
        self.model.eval()
        with torch.no_grad():
            if(self.machine=="gpu"):
                output = self.model(self.features[self.idx_test].cuda(), self.adj[self.idx_test][:, self.idx_test].cuda())
                test_loss = F.nll_loss(output, self.labels[self.idx_test].cuda())
                test_acc = accuracy(output, self.labels[self.idx_test].cuda())
            else:
                output = self.model(self.features[self.idx_test],
                                    self.adj[self.idx_test][:, self.idx_test])
                test_loss = F.nll_loss(output, self.labels[self.idx_test])
                test_acc = accuracy(output, self.labels[self.idx_test])
            print("Test set results_1hop:", "loss= {:.4f}".format(test_loss.item()),
                  "accuracy= {:.4f}".format(test_acc.item()))

            return test_loss.item(), test_acc.item()

machine = "cpu"
dataset = "citeseer"
alpha = 0
batch_size = 30
# presample_disk = 5
# retrieve_cache = 3
# retrieve_disk = 4
# erase_cache = 2
disk_read = 3
sampled_depth = 2
task = "KHhopGCN" #"KHhopSGC"

# test_loss_list = []
# test_acc_list = []
# train_time_list = []
# i_list = []
#
# start_time = time.time()
# Creating a trainer instance
trainer = Trainer(machine,dataset)

# Train the model
trainer.train()

# end_time = time.time()
# train_time = end_time - start_time

# Test the model
test_loss, test_acc = trainer.test()

# test_loss_list.append(test_loss)
# test_acc_list.append(test_acc)
# train_time_list.append(train_time)
#
# draw_pic(i_list, test_loss_list, "Test Loss versus Alpha Changes","red","alpha","test loss")
# draw_pic(i_list, test_acc_list, "Test Accuracy versus Alpha Changes","blue","alpha","test accuracy")
# draw_pic(i_list, train_time_list, "Training Time versus Alpha Changes","green","alpha","Training Time")
