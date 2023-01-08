import dgl
import torch
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph


# TODO:
#  1. Learn how to handle HeteroGraph.
#  2. Learn how to pre-fetch the node features in sampling.
#  3. Learn how to do sampling in GPU.

class CustomGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, out_feats)

    def forward(self, g: DGLGraph, h):
        """
        Note that:
            1. If the input feature is a pair of tensors, then the input graph must be unidirectional bipartite.
            2. If the input feature is a single tensor and the input graph is a MFG, DGL will automatically set the feature on the output nodes as the first few rows of the input node features.
            3. If the input feature must be a single tensor and the input graph is not a MFG, then the input graph must be homogeneous.
        """

        with g.local_scope():

            if isinstance(h, tuple):
                h_src, h_dst = h
            elif g.is_block:
                h_src = h
                h_dst = h[:g.num_dst_nodes()]
            else:
                h_src = h_dst = h

            # print(f"[debug]: g.srcnodes().shape: {g.srcnodes().shape}")
            # print(f"[debug]: h_src.shape: {h_src.shape}")

            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            # return self.W(g.dstdata['h_neigh'])
            return self.W(torch.cat([g.dstdata['h'], g.dstdata['h_neigh']], 1))


# use dgl's implemented GNN layer for mini-batch training
class CustomTwoLayerGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.n_layers = 2
        self.conv1 = CustomGraphConv(in_feats, hid_feats)
        self.conv2 = CustomGraphConv(hid_feats, out_feats)

    def forward(self, blocks, x):
        h = F.relu(self.conv1(blocks[0], x))
        h = F.relu(self.conv2(blocks[1], h))
        return h

    def inference(self, g: DGLGraph, x, batch_size, device):
        """
        Offline inference with this module

        Note that: in offline inference, we compute representation layer by layer
        """

        # compute representation layer by layer
        for l, layer in enumerate([self.conv1, self.conv2]):
            y = torch.zeros(g.num_nodes(), self.hid_feats if l != self.n_layers - 1 else self.out_feats)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(g, torch.arange(g.num_nodes()), sampler, batch_size=batch_size,
                                                        shuffle=True, drop_last=False)

            # within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                h = x[input_nodes].to(device)
                h_dst = h[:block.num_dst_nodes()]
                h = F.relu(layer(block, (h, h_dst)))
                y[output_nodes] = h.cpu()

            x = y

        _, indices = torch.max(y, dim=1)

        return indices


# TODO: learn how to write sampler for heterogeneous graphs and write edge-wise sampler, which can be converted from
#  node-wise sampler by calling dgl.dataloading.as_edge_prediction_sampler().
class CustomNeighborSampler(dgl.dataloading.Sampler):
    def __init__(self, fanouts: list[int]):
        super().__init__()
        self.fanouts = fanouts

    def sample(self, g: DGLGraph, seed_nodes):
        output_nodes = seed_nodes
        subgs = []
        for fanout in reversed(self.fanouts):
            # Sample a fixed number of neighbors of the current seed nodes.
            sg = g.sample_neighbors(seed_nodes, fanout)

            # Convert this subgraph to a message flow graph.
            sg = dgl.to_block(sg, seed_nodes)
            seed_nodes = sg.srcdata[dgl.NID]
            subgs.insert(0, sg)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, subgs

class ReversedNeighborSampler(dgl.dataloading.Sampler):
    """
    Unlike existing sampler, which samples subgraphs by gradually extending few center nodes to outer nodes,
    ReversedNeighborSampler does the sampling in reverse direction, it tries to sample many outer nodes at first.
    Then it gradually samples towards center nodes, whose number are smaller than outer nodes.
    """
    def __init__(self, dropout: list[float], num_seeds: int, prob: str = "in_degree", mask: str = "train_mask"):
        super().__init__()
        self.dropout = dropout
        self.num_seeds = num_seeds
        self.prob = prob
        self.mask = mask

    def sample(self, g: DGLGraph, seed_nodes):
        nodes = g.nodes()
        seed_nodes = nodes[torch.randint(nodes.shape[0], (self.num_seeds,))]
        input_nodes = seed_nodes
        cur_prob = self.prob
        subgs = []

        for idx, dropout in enumerate(self.dropout):
            # For each seed node, we drop out some of them according to `dropout`, and sample only one of its
            # neighbor based on in_degree.
            # survived_nodes = seed_nodes[torch.randint(len(seed_nodes), (int(len(seed_nodes) * dropout),))] // some problem with dropout
            survived_nodes = seed_nodes
            sg = g.sample_neighbors(survived_nodes, 1, prob=cur_prob, edge_dir='out')

            # print(f"[debug]: before dgl.to_block:\n survived_nodes:{survived_nodes},\n sg:{sg}, sg.edges:{sg.edges()},
            # sg.nodes:{sg.nodes()}")

            # Convert this subgraph to a message flow graph
            # print(f"[debug]: sg.srcnodes().shape:{sg.srcnodes().shape}")
            # print(f"[debug]: seed.shape:{seed_nodes.shape}")
            # srcnodes, dstnodes = sg.edges()
            # print(f"[debug]: {srcnodes.shape}, {dstnodes.shape}")

            sg = dgl.to_block(sg, dgl.to_block(sg).dstdata[dgl.NID]) # TODO: Fix the bug here.
            seed_nodes = sg.dstdata[dgl.NID]
            # print(f"[debug]: after dgl.to_block:\n sg.srcnodes():{sg.srcnodes()}, sg.srcdata():{sg.srcdata[dgl.NID]},
            # sg.dstnodes():{sg.dstnodes()}, sg.dstdata():{sg.dstdata[dgl.NID]}")
            subgs.append(sg)

        output_nodes = subgs[-1].dstnodes()
        return input_nodes, output_nodes, subgs


def run_stochastic_training():

    def prepare_dataloader(g: DGLGraph, sampler: dgl.dataloading.Sampler):
        node_features = g.ndata['feat']
        node_labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        valid_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        n_features = node_features.shape[1]
        n_labels = int(node_labels.max().item() + 1)

        train_nids = g.nodes()[train_mask]
        valid_nids = g.nodes()[valid_mask]
        test_nids = g.nodes()[test_mask]

        dgl.dataloading.DataLoader

        train_dataloader = dgl.dataloading.DataLoader(
            g, train_nids, sampler, batch_size=16, shuffle=True, drop_last=False, num_workers=2
        )
        valid_dataloader = dgl.dataloading.DataLoader(
            g, valid_nids, sampler, batch_size=16, shuffle=True, drop_last=False, num_workers=2
        )
        test_dataloader = dgl.dataloading.DataLoader(
            g, test_nids, sampler, batch_size=16, shuffle=True, drop_last=False, num_workers=2
        )

        return n_features, n_labels, train_dataloader, valid_dataloader, test_dataloader


    def evaluate(model, valid_dataloader: dgl.dataloading.DataLoader, maskStr:str):
        model.eval()
        correct = 0
        num_valid = 0
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in valid_dataloader:
                input_graphs = [b.to(torch.device('cpu')) for b in blocks]
                mask = input_graphs[-1].dstdata[maskStr]
                num_valid = num_valid + mask.nonzero().shape[0]
                input_features = input_graphs[0].srcdata['feat']
                output_labels = input_graphs[-1].dstdata['label'][mask]
                logits = model(input_graphs, input_features)[mask]
                _, indices = torch.max(logits, dim=1)
                correct += torch.sum(indices == output_labels).item()
        return correct * 1.0 / num_valid


    def train_epoch(model, opt, train_dataloader: dgl.dataloading.DataLoader):
        model.train()
        for input_nodes, output_nodes, blocks in train_dataloader:

            print(f"[debug]: blocks:{blocks}")

            input_graphs = [b.to(torch.device('cpu')) for b in blocks]
            input_features = input_graphs[0].srcdata['feat']

            # print(f"[debug]: {input_features.shape}")
            # print(f"[debug]: {input_graphs[0].srcnodes().shape}")

            output_labels = input_graphs[-1].dstdata['label']
            train_mask = input_graphs[-1].dstdata['train_mask']
            logits = model(input_graphs, input_features)
            loss = F.cross_entropy(logits[train_mask], output_labels[train_mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss


    def node_classification(train_dataloader, valid_dataloader, test_dataloader, n_features, n_labels, num_epoch):
        model = CustomTwoLayerGNN(in_feats=n_features, hid_feats=100, out_feats=n_labels)
        opt = torch.optim.Adam(model.parameters())

        for epoch in range(num_epoch):
            loss = train_epoch(model, opt, train_dataloader)
            valid_acc = evaluate(model, valid_dataloader, "val_mask")

            print(f"epoch:{epoch}, loss:{loss}, valid_acc:{valid_acc}")

        # test_acc = evaluate(model, test_dataloader, "test_mask")
        # print(f"test_acc:{test_acc}"}

        return model

    dataset = dgl.data.CiteseerGraphDataset()
    citeseer_graph: DGLGraph = dataset[0]
    g = citeseer_graph
    # sampler = CustomNeighborSampler([10, 10])  # use our custom sampler
    sampler = ReversedNeighborSampler([0.85, 0.85], max(g.nodes().shape[0] // 5, 8))
    n_features, n_labels, train_dataloader, valid_dataloader, test_dataloader = prepare_dataloader(g,
                                                                                                   sampler)
    model = node_classification(train_dataloader, valid_dataloader, test_dataloader, n_features, n_labels, 5)
    # y = model.inference(citeseer_graph, citeseer_graph.ndata['feat'], 256, torch.device('cpu'))
    #
    # print(f"inference results: {y}")


def test_neighbor_sampler():
    sampler = CustomNeighborSampler([1, 2])
    u = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    v = torch.tensor([1, 2, 3, 2, 3, 0, 0, 1, 3, 0, 1, 2])
    g = dgl.graph((u, v))
    seed_nodes = [2]

    input_nodes, output_nodes, subgs = sampler.sample(g, seed_nodes)

    print(f"input_nodes:{input_nodes}, output_nodes:{output_nodes}, subgs:{subgs}")
    graphs = [b.edges() for b in subgs]
    print(f"graphs:{graphs}")

def test_reversed_neighbor_sampler():

    # Custom Graph
    u = torch.tensor([1, 2, 3, 4, 5, 6, 7, 7, 8])
    v = torch.tensor([7, 7, 7, 7, 8, 8, 8, 9, 9])
    g:DGLGraph = dgl.graph((u, v))
    g.ndata['train_mask'] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # Citeseer Graph
    dataset = dgl.data.CiteseerGraphDataset()
    citeseer_graph: DGLGraph = dataset[0]
    g = citeseer_graph

    # label each edge by its source node's degree.
    deg = g.in_degrees()
    train_mask = g.ndata['train_mask'].int()
    srcnodes, dstnodes = g.edges()
    g.edata['in_degree'] = deg[srcnodes].double()
    g.edata['train_mask'] = train_mask[dstnodes].double()

    # print(f"g:{g}, srcnodes:{srcnodes}, dstnodes:{dstnodes}, in_degree:{deg}, e_degree:{e_degree}")

    # perform sampling
    sampler = ReversedNeighborSampler([0.85, 0.85], max(g.nodes().shape[0] // 5, 8))
    input_nodes, output_nodes, subgs = sampler.sample(g)

    # print(f"input_nodes:{input_nodes}, output_nodes:{output_nodes}")
    for sg in subgs:
        # print(f"sg:{sg}, sg.edges():{sg.edges()} sg.srcnodes():{sg.srcnodes()}, sg.src_origin():{sg.srcdata[dgl.NID]}"
        #       f", sg.dstnodes():{sg.dstnodes()}, sg.dst_origin():{sg.dstdata[dgl.NID]},"
        #       f" sg.dst_mask():{sg.dstdata['train_mask']}")
        print(f"sg:{sg}, sg.dst_mask:{sg.dstdata['train_mask']}")




if __name__ == "__main__":
    # test_reversed_neighbor_sampler()

    # TODO: (1) we need to add self edge from source node to destination node in block, currently there is bug.
    run_stochastic_training()