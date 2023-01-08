import dgl
import torch

"""
Distributed Training

System components:
1. KV stores: storing the graph structure, node features, and edge features.
2. Sampler: for creating mini-batches of subgraphs for training.
3. Trainer: for training the network using sampled subgraphs.

Distributed APIs:
1. Distributed Graph: storing partitioned graph
2. Distributed Tensor: storing partitioned tensor of node features
3. Distributed DisEmbedding: ???
4. DistDataLoader: loading data in mini-batch distributively
    a. DisDataLoader
    b. DisNodeDataLoader
    c. DisEdgeDataLoader

Steps:
1. Partition the graph
2. Load the partitioned graph
3. Perform sampling
4. Perform training
"""