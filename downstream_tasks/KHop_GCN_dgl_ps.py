import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import dgl
print(dgl.__version__)
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from ogb.nodeproppred import DglNodePropPredDataset

# Load the ogbn-arxiv dataset
dataset = DglNodePropPredDataset(name='ogbn-arxiv')
g, labels = dataset[0]  # Graph and labels
g.ndata['label'] = labels[:, 0]  # Adjust labels shape

# Add self-loop since GCN model assumes self-loops in the graph
g = dgl.add_self_loop(g)

# Extract features and labels
features = g.ndata['feat']
labels = g.ndata['label']

# Get the split indices from the dataset
split_idx = dataset.get_idx_split()
train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

# Convert indices to boolean masks
train_mask = torch.zeros(labels.shape[0], dtype=torch.bool).scatter_(0, train_idx, True)
val_mask = torch.zeros(labels.shape[0], dtype=torch.bool).scatter_(0, val_idx, True)
test_mask = torch.zeros(labels.shape[0], dtype=torch.bool).scatter_(0, test_idx, True)

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, blocks, x):
        h = self.conv1(blocks[0], x)
        h = F.relu(h)
        h = self.conv2(blocks[1], h)
        return h

# Initialize the model and the optimizer
model = GCN(features.shape[1], 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Function to compute accuracy
def compute_accuracy(logits, labels, mask):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / sum(mask)

# Define the sampler and dataloader for training
sampler = MultiLayerNeighborSampler([100, 100])
train_dataloader = NodeDataLoader(
    g,
    train_idx,
    sampler,
    batch_size=128,
    shuffle=True,
    drop_last=False,
    num_workers=0
)

# Evaluation function
def evaluate(model, dataloader, labels, mask):
    model.eval()
    total_correct = 0
    total = 0
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for b in blocks]
        input_features = blocks[0].srcdata['feat']
        output_labels = labels[output_nodes]
        logits = model(blocks, input_features)
        _, predicted = torch.max(logits, dim=1)
        total_correct += (predicted == output_labels).sum().item()
        total += len(output_labels)
    return total_correct / total

# Training loop
def train(epochs=100):
    best_val_acc = 0
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_nodes, output_nodes, blocks in train_dataloader:
            blocks = [b.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = labels[output_nodes]
            logits = model(blocks, input_features)
            loss = F.cross_entropy(logits, output_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_dataloader = NodeDataLoader(
            g,
            val_idx,
            sampler,
            batch_size=128,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        val_acc = evaluate(model, val_dataloader, labels, val_mask)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataloader)}, Val Acc: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # Restore the best model state
    model.load_state_dict(best_model_state)

train()

# Testing
test_dataloader = NodeDataLoader(
    g,
    test_idx,
    sampler,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    num_workers=0
)
test_acc = evaluate(model, test_dataloader, labels, test_mask)
print(f"Test Acc: {test_acc}")
