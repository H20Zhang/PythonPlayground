import dgl
import torch
from dgl import DGLGraph
import pandas as pd


if __name__ == "__main__":

    # default method for creating graph
    u = torch.tensor([0, 0, 0, 1])
    v = torch.tensor([1, 2, 3, 3])
    g1: DGLGraph = dgl.graph((u, v))
    print(f"g1.edges():{g1.edges()}")

    # create graph with isolated graph
    g2 = dgl.graph((u, v), num_nodes=8)  # nodes with id 0-7
    print(f"g2.edges():{g2.edges()}")

    # create undirected graph with bidirectional edges
    g3 = dgl.to_bidirected(g2)
    print(f"g3.edges():{g3.edges()}")

    # create heterograph with 3 node type and 3 edges types
    graph_data = {('drug', 'interacts', 'drug'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
                  ('drug', 'interacts', 'gene'): (torch.tensor([0, 1]), torch.tensor([2, 3])),
                  ('drug', 'treats', 'disease'): (torch.tensor([1]), torch.tensor([2]))}

    g5: DGLGraph = dgl.heterograph(graph_data)


    # accessing node and edge properties
    print(f"g1.nodes():{g1.nodes()}")
    print(f"g1.edges():{g1.edges()}")
    print(f"g5.ntypes:{g5.ntypes}, g5.etypes:{g5.etypes}")
    print(f"g5.nodes('drug'):{g5.nodes('drug')}")
    print(f"g5.num_nodes('drug'):{g5.num_nodes('drug')}")

    g1.ndata['x'] = torch.ones(g1.num_nodes(), 3)  # only numeric types are allows in ndata and edata
    g1.edata['w'] = torch.ones(g1.num_edges(), dtype=torch.int32)
    g1.ndata['y'] = torch.randn(g1.num_nodes(), 5)
    g5.nodes['drug'].data['hv'] = torch.ones(g5.num_nodes('drug'), 1)
    g5.edges["treats"].data['x'] = torch.ones(1, 1)  # note: 'interacts' edges cannot be assigned this way, as its
    # source and destination nodes have two types

    print(f"x property of node 1: {g1.ndata['x'][1]}")
    print(f"weight of edge 1: {g1.edata['w'][1]}")

    home_dir = '../DGL_PlayGround'

    # load graph from CSV files
    nodes_data = pd.read_csv(home_dir+'/data/demo_graph/nodes.csv')
    edges_data = pd.read_csv(home_dir+'/data/demo_graph/edges.csv')

    print(nodes_data)
    print(edges_data)

    src = edges_data['src_id'].to_numpy()
    dst = edges_data['dst_id'].to_numpy()

    g4: DGLGraph = dgl.graph((src, dst))
    g4.ndata['age'] = torch.from_numpy(nodes_data['age'].to_numpy())  # assign data to node property

    print(g4)
