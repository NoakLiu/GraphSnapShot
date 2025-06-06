import torch
import torch.optim as optim
from data_processing.data_preprocessing import load_data
import torch.nn.functional as F
from data_processing.data_preprocessing import accuracy
from SSDReS_Samplers.GraphSDSampler import GraphSDSampler, MLP
import numpy as np
import time
from data_processing.plot import draw_pic


class Trainer:
    def __init__(self, machine, dataset):
        self.machine = machine
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data(dataset=dataset)
        if(self.machine == "gpu"):
            self.model = MLP(self.features.size(1), max(self.labels) + 1).cuda()
        else:
            self.model = MLP(self.features.size(1), max(self.labels) + 1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self):

        # Training
        sampler = GraphSDSampler(list(range(self.adj.size(0))), presampled_nodes) #[self.idx_train]

        for epoch in range(100):
            self.model.train()
            self.optimizer.zero_grad()

            # if this is because the data loading module
            # Sampling nodes
            sampled_nodes = sampler.resample(resampled_nodes, alpha=alpha, mode="exchange")
            sampled_adj = self.adj[sampled_nodes][:, sampled_nodes]
            sampled_features = self.features[sampled_nodes]
            sampled_labels = self.labels[sampled_nodes]

            # Forward
            if(self.machine=="gpu"):
                output = self.model(sampled_features.cuda(), sampled_adj.cuda())
                loss = F.nll_loss(output, sampled_labels.cuda())
            else:
                output = self.model(sampled_features, sampled_adj)
                loss = F.nll_loss(output, sampled_labels)

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
                print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss.item()),
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
presampled_nodes = 50
resampled_nodes = 40

test_loss_list = []
test_acc_list = []
train_time_list = []
i_list = []

for i in np.arange(0.1,1,0.1):
    alpha = i

    start_time = time.time()
    # Creating a trainer instance
    trainer = Trainer(machine,dataset)

    print(alpha)

    # Train the model
    trainer.train()

    end_time = time.time()
    train_time = end_time - start_time

    # Test the model
    test_loss, test_acc = trainer.test()

    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    train_time_list.append(train_time)
    i_list.append(i)

draw_pic(i_list, test_loss_list, "Test Loss versus Alpha Changes","red","alpha","test loss")
draw_pic(i_list, test_acc_list, "Test Accuracy versus Alpha Changes","blue","alpha","test accuracy")
draw_pic(i_list, train_time_list, "Training Time versus Alpha Changes","green","alpha","Training Time")

