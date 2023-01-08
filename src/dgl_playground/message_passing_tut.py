import torch
import dgl
from dgl import DGLGraph
from dgl.udf import EdgeBatch, NodeBatch
import dgl.function as fn

# TODO: finish the Chapter 2.5 (message passing on heterogeneous graph) of the tutorial
#  "https://docs.dgl.ai/guide/message.html"
if __name__ == '__main__':

    u = torch.tensor([0, 1, 2])
    v = torch.tensor([1, 2, 3])
    g: DGLGraph = dgl.graph((u, v))

    # The naming convention for message built-in funcs is that u represents src nodes, v represents dst nodes,
    # and e represents edges. The parameters for those functions are strings indicating the input and output field
    # names for the corresponding nodes and edges.

    # try to use built-in message_func and reduce_func as much as possible

    # custom message passing API
    def message_func(edges: EdgeBatch):
        return {'he': edges.src['hu'] + edges.dst['hv']}

    def reduce_func(nodes: NodeBatch):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

    '''
    g.apply_edges(fn.u_add_v('el', 'er', 'e')), it is possibile to invoke edge-wise computation by apply edges
    without invoking message passing
    '''

    '''graph.update_all performs a message passing, in which the message_func generates the messages, reduce_func 
    reduces the messages generated, update_func updates the feature of nodes 
    
    def update_all_example(graph):
        # store the result in graph.ndata['ft']
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        # Call update function outside of update_all, which is more straight-forward
        final_ft = graph.ndata['ft'] * 2
        return final_ft
    '''

    '''
    Recommendation for writing efficient message passing code:
    1. utilize built-in message and reduce functions as much as possible.
    2. try not to store too much data on the edge properties of the graph, which can be huge.
    3. try to find equivalent form of the message passing code, which is faster.
    '''

    '''
    Apply message passing on part of the graph, which is very common in mini-batch training
    
    nid = [0, 2, 3, 4, 5, 6]
    sg = g.subgraph(nid)
    sg.update_all(message_func, reduce_func, apply_node_func())
    '''
