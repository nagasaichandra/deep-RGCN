import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin
from utils import *
import torch_geometric as tg


class MLP(Seq):
    """
    Multi Layer Perceptron
    """
    def __init__(self, channels, act='relu',
                 norm=None, bias=True,
                 drop=0., last_lin=False):
        m = []

        for i in range(1, len(channels)):

            m.append(Lin(channels[i - 1], channels[i], bias))

            if (i == len(channels) - 1) and last_lin:
                pass
            else:
                if norm is not None and norm.lower() != 'none':
                    m.append(norm_layer(norm, channels[i]))
                if act is not None and act.lower() != 'none':
                    m.append(act_layer(act))
                if drop > 0:
                    m.append(nn.Dropout2d(drop))

        self.m = m
        super(MLP, self).__init__(*self.m)


class MultiSeq(Seq):
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class RGCN(nn.Module):

    """
    RGCN block using RGCNConv from torch_geometric
    """
    def __init__(self, in_channels, out_channels, num_relations, act='relu', norm=None, bias=True, aggr='mean',
                 num_bases=None, num_blocks=None):
        super(RGCN, self).__init__()
        self.rgcn = tg.nn.RGCNConv(in_channels, out_channels, num_relations, num_bases, num_blocks, aggr, bias)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.post_layer = nn.Sequential(*m)

    def forward(self, x, edge_index, edge_type):
        x = self.rgcn(x, edge_index, edge_type)
        out = self.post_layer(x)
        return out


class Plain(nn.Module):
    """
     Plain graph convolution block
    """
    def __init__(self, in_channels, out_channels, num_relations, act='relu', norm=None, bias=True, aggr='mean',
                 num_bases=None, num_blocks=None):
        super(Plain, self).__init__()
        self.body = RGCN(in_channels, out_channels, num_relations, act, norm, bias, aggr, num_bases, num_blocks)

    def forward(self, x, edge_index, edge_type):
        return self.body(x, edge_index, edge_type), edge_index


class DenseGraphBlock(nn.Module):
    """
    Dense Static graph convolution block
    """
    def __init__(self, in_channels,  out_channels, num_relations, act='relu', norm=None, bias=True, aggr='mean',
                 num_bases=None, num_blocks=None):
        super(DenseGraphBlock, self).__init__()
        self.body = RGCN(in_channels, out_channels, num_relations, act, norm, bias, aggr, num_bases, num_blocks)

    def forward(self, x, edge_index, edge_type):
        out = self.body(x, edge_index, edge_type)
        return torch.cat((x, out), 1), edge_index


class ResGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, in_channels, out_channels, num_relations, act='relu', norm=None, bias=True, aggr='mean',
                 num_bases=None, num_blocks=None, res_scale=1):
        super(ResGraphBlock, self).__init__()
        self.body = RGCN(in_channels, out_channels, num_relations, act, norm, bias, aggr, num_bases, num_blocks)
        self.res_scale = res_scale

    def forward(self, x, edge_index, edge_type):
        out = self.body(x, edge_index, edge_type)
        return out + x*self.res_scale, edge_index
