from blocks import *
from utils import *
import torch.nn.functional as F


class DenseR_no_fusion(torch.nn.Module):
    def __init__(self, opt):
        super(DenseR_no_fusion, self).__init__()
        self.channels = opt.channels
        self.n_layers = opt.n_layers
        self.act = opt.act
        self.norm = opt.norm
        self.bias = opt.bias
        self.block_type = opt.block_type
        self.relations = opt.relations
        self.decomposition = opt.dec
        self.n_blocks = 30 if self.decomposition.lower() == 'block' else None
        self.n_bases = opt.bases if self.decomposition.lower() == 'basis' else None
        self.aggr = opt.aggr
        self.dropout = opt.dropout
        self.c_growth = 0
        self.res_scale = 0
        self.conv1 = RGCN(opt.num_nodes, self.channels, self.relations, act=self.act, norm=self.norm,
                          bias=self.bias, aggr='mean', num_bases=self.n_bases, num_blocks=self.n_blocks)

        if self.block_type.lower() == 'dense':
            self.c_growth = self.channels
            self.backbone = MultiSeq(
                *[DenseGraphBlock(self.channels + i * self.c_growth, self.channels, num_relations=self.relations,
                                  act=self.act, norm=self.norm, bias=self.bias, aggr=self.aggr, num_bases=self.n_bases,
                                  num_blocks=self.n_blocks)
                  for i in range(self.n_layers - 1)])

        elif self.block_type.lower() == 'res':
            self.res_scale = 1
            self.backbone = MultiSeq(*[
                ResGraphBlock(self.channels, self.channels, num_relations=self.relations, act=self.act, norm=self.norm,
                              bias=self.bias, aggr=self.aggr, num_bases=self.n_bases, num_blocks=self.n_blocks,
                              res_scale=self.res_scale)
                for _ in range(self.n_layers - 1)])

        elif self.block_type.lower() == 'plain':
            self.backbone = MultiSeq(
                *[Plain(self.channels, self.channels, num_relations=self.relations, act=self.act, norm=self.norm,
                        bias=self.bias, aggr=self.aggr, num_bases=self.n_bases, num_blocks=self.n_blocks)
                  for _ in range(self.n_layers - 1)])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.xavier_uniform_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, edge_index, edge_type):
        features = [self.conv1(None, edge_index, edge_type)]
        for i in range(self.n_layers - 1):
            features.append(self.backbone[i](features[-1], edge_index, edge_type)[0])
        return F.log_softmax(features[-1])
