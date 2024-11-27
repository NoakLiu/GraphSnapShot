# import dgl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn import GraphConv
# from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset

# # Load the ogbn-arxiv dataset
# dataset = DglNodePropPredDataset(name='ogbn-arxiv')
# g, labels = dataset[0]  # Graph and labels
# g.ndata['label'] = labels[:, 0]  # Adjust labels shape

# # Add self-loop since GCN model assumes self-loops in the graph
# g = dgl.add_self_loop(g)

# # Extract features and labels
# features = g.ndata['feat']
# labels = g.ndata['label']

# # Get the split indices from the dataset
# split_idx = dataset.get_idx_split()
# train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

# # Convert indices to boolean masks
# train_mask = torch.zeros(labels.shape[0], dtype=torch.bool).scatter_(0, train_idx, True)
# val_mask = torch.zeros(labels.shape[0], dtype=torch.bool).scatter_(0, val_idx, True)
# test_mask = torch.zeros(labels.shape[0], dtype=torch.bool).scatter_(0, test_idx, True)

# # Define the GCN model
# class GCN(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, num_classes)

#     def forward(self, blocks, x):
#         h = self.conv1(blocks[0], x)
#         h = F.relu(h)
#         h = self.conv2(blocks[1], h)
#         return h

# # Initialize the model and the optimizer
# model = GCN(features.shape[1], 16, dataset.num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# # Function to compute accuracy
# def compute_accuracy(logits, labels, mask):
#     _, indices = torch.max(logits, dim=1)
#     correct = torch.sum(indices[mask] == labels[mask])
#     return correct.item() * 1.0 / sum(mask)

# # Define the sampler and dataloader for training
# # sampler = MultiLayerNeighborSampler([15, 10])

# sampler = MultiLayerFullNeighborSampler(
#     1, prefetch_node_feats=["feat"], fused=fused_sampling
# )
# train_dataloader = NodeDataLoader(
#     g,
#     train_idx,
#     sampler,
#     batch_size=128,
#     shuffle=True,
#     drop_last=False,
#     num_workers=0
# )

# # Evaluation function
# def evaluate(model, dataloader, labels, mask):
#     model.eval()
#     total_correct = 0
#     total = 0
#     for input_nodes, output_nodes, blocks in dataloader:
#         blocks = [b.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for b in blocks]
#         input_features = blocks[0].srcdata['feat']
#         output_labels = labels[output_nodes]
#         logits = model(blocks, input_features)
#         _, predicted = torch.max(logits, dim=1)
#         total_correct += (predicted == output_labels).sum().item()
#         total += len(output_labels)
#     return total_correct / total

# # Training loop
# def train(epochs=100):
#     best_val_acc = 0
#     best_model_state = None
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for input_nodes, output_nodes, blocks in train_dataloader:
#             blocks = [b.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for b in blocks]
#             input_features = blocks[0].srcdata['feat']
#             output_labels = labels[output_nodes]
#             logits = model(blocks, input_features)
#             loss = F.cross_entropy(logits, output_labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         val_dataloader = NodeDataLoader(
#             g,
#             val_idx,
#             sampler,
#             batch_size=128,
#             shuffle=False,
#             drop_last=False,
#             num_workers=0
#         )
#         val_acc = evaluate(model, val_dataloader, labels, val_mask)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataloader)}, Val Acc: {val_acc}")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_model_state = model.state_dict()

#     # Restore the best model state
#     model.load_state_dict(best_model_state)

# train()

# # Testing
# test_dataloader = NodeDataLoader(
#     g,
#     test_idx,
#     sampler,
#     batch_size=128,
#     shuffle=False,
#     drop_last=False,
#     num_workers=0
# )
# test_acc = evaluate(model, test_dataloader, labels, test_mask)
# print(f"Test Acc: {test_acc}")

import argparse
import time

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from dgl.nn import GraphConv

@torch.no_grad()
def layerwise_infer(
    device, graph, nid, model, num_classes, batch_size, fused_sampling
):
    model.eval()
    pred = model.inference(
        graph, device, batch_size, fused_sampling
    )  # pred in buffer_device.
    pred = pred[nid]
    label = graph.ndata["label"][nid].to(pred.device)
    return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)

@torch.no_grad()
def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        x = blocks[0].srcdata["feat"]
        ys.append(blocks[-1].dstdata["label"])
        y_hat = model(blocks, x)
        y_hats.append(y_hat.argmax(dim=1))  # Take the class with the highest probability
    preds = torch.cat(y_hats, dim=0)
    target = torch.cat(ys, dim=0)
    return MF.accuracy(preds, target, task="multiclass", num_classes=num_classes)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_size, hidden_size))
        self.layers.append(GraphConv(hidden_size, out_feats))

    def forward(self, blocks, x):
        for layer, block in zip(self.layers, blocks):
            x = layer(block, x)
            x = F.relu(x)
        return x

    def inference(self, g, device, fused_sampling: bool = True):
        with g.local_scope():
            for layer in self.layers:
                h = g.ndata['h'].to(device)
                g.ndata['h'] = layer(g, h)
                g.apply_edges(fn.copy_u('h', 'm'), etype=None)
                g.update_all(fn.mean('m', 'h'), fn.sum('h', 'neigh'))
            return g.ndata.pop('neigh')

def train(device, g, dataset, model, num_classes, use_uva):
    train_idx = dataset.train_idx.to(g.device if not use_uva else device)
    val_idx = dataset.val_idx.to(g.device if not use_uva else device)

    # sampler = MultiLayerNeighborSampler([10] * (num_layers - 1))
    sampler = MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["feat"], fused=fused_sampling
        )
    train_dataloader = DataLoader(
        g, train_idx, sampler, device=device, batch_size=1024, shuffle=True
    )
    val_dataloader = DataLoader(
        g, val_idx, sampler, device=device, batch_size=1024, shuffle=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        t0 = time.time()
        model.train()
        total_loss = 0
        for input_nodes, output_nodes, blocks in train_dataloader:
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        t1 = time.time()
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(f"Epoch {epoch:05d} | Loss {total_loss / len(train_dataloader):.4f} | Accuracy {acc.item():.4f} | Time {t1 - t0:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mixed", choices=["cpu", "mixed", "gpu"], help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, 'gpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]
    
    # Add self-loops to the graph
    g = dgl.add_self_loop(g)
    
    if args.mode != "cpu":
        g = g.to('cuda')
    num_classes = dataset.num_classes
    use_uva = args.mode == "mixed"
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    num_layers = 3  # Number of layers in the GCN
    model = GCN(g.ndata['feat'].shape[1], 256, num_classes, num_layers).to(device)

    print("Training...")
    train(device, g, dataset, model, num_classes, use_uva)

    print("Testing...")
    acc = layerwise_infer(
        device,
        g,
        dataset.test_idx,
        model,
        num_classes,
        batch_size=4096,
        fused_sampling=fused_sampling
    )
    print(f"Test accuracy {acc.item():.4f}")


