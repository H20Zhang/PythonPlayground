import dgl
import torch.nn as nn
import torch.nn.functional as F
import building_gnn_tut
import torch

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = building_gnn_tut.SAGEConv(in_feats, hid_feats, aggregator_type='gcn')
        self.conv2 = building_gnn_tut.SAGEConv(hid_feats, out_feats, aggregator_type='gcn')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = dgl.data.CoraGraphDataset()
    citeseer_graph = dataset[0].to(device)

    node_features = citeseer_graph.ndata['feat']
    node_labels = citeseer_graph.ndata['label']
    train_mask = citeseer_graph.ndata['train_mask']
    valid_mask = citeseer_graph.ndata['val_mask']
    test_mask = citeseer_graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)

    print(f"n_features:{n_features}, n_labels:{n_labels}")

    def evaluate(model, graph, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def node_classification(graph, num_epoch):
        model = SAGE(in_feats=n_features, hid_feats=128, out_feats=n_labels)
        model.to(device)
        opt = torch.optim.Adam(model.parameters())

        for epoch in range(num_epoch):
            model.train()
            logits = model(graph, node_features)
            loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
            acc = evaluate(model, graph, node_features, node_labels, valid_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"epoch:{epoch}, loss:{loss}, acc:{acc}")

    node_classification(citeseer_graph, 200)

