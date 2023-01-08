import torch
import dgl
import torch.nn as nn
from dgl import DGLGraph
import dgl.function as fn
import torch.nn.functional as F
from dgl.utils import expand_as_pair
from dgl.utils import check_eq_shape

# TODO: read the chapter 3.3 (Heterogeneous GraphConv Module) https://docs.dgl.ai/guide/nn-heterograph.html

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 bias=True,
                 norm=None,
                 activation=None
                 ):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.activation = activation

        if aggregator_type not in ['mean', 'pool', 'lstm', 'gcn']:
            raise KeyError(f'Aggregator type {aggregator_type} not supported.')

        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_dst_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type in ['mean', 'pool', 'lstm']:
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph: DGLGraph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = feat_dst
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            else:
                raise KeyError(f"Aggregator type {self._aggre_type} not recognized.")

            # GraphSageGCN does ont require fc_self
            # if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
            # else:
            #     rst = self.fc_self(h_self) + self.fc_neigh(h_neigh) #TODO: this part has some problem regarding h_self

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst













