import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
# from sklearn.metrics import f1_score
from architecture import DenseR
from args import OptInit
# from utils import save_checkpoint
# from utils.metrics import AverageMeter
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import Entities
import statistics as s
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, data.edge_type)
    loss = F.nll_loss(out[train_idx], train_y)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def valid():
    model.eval()

    out = model(data.edge_index, data.edge_type)
    pred = out.argmax(dim=-1)
    valid_loss = F.nll_loss(out[valid_idx], valid_y)
    train_acc = pred[train_idx].eq(train_y).to(torch.float).mean()
    valid_acc = pred[valid_idx].eq(valid_y).to(torch.float).mean()

    return valid_loss.item(), train_acc.item(), valid_acc.item()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.edge_index, data.edge_type)
    pred = out.argmax(dim=-1)
    train_acc = pred[train_idx].eq(train_y).to(torch.float).mean()
    test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()
    return train_acc.item(), test_acc.item()


if __name__ == '__main__':
    opt = OptInit().initialize()
    opt.printer.info(' === Downloading Dataset and Creating graphs ===')
    opt.dataset = Entities(opt.data_dir, opt.dataset)
    print('done')
    opt.data = opt.dataset[0]
    node_idx = torch.cat([opt.data.train_idx, opt.data.test_idx], dim=0)
    node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, 2, opt.data.edge_index, relabel_nodes=True)

    opt.data.num_nodes = node_idx.size(0)
    opt.data.edge_index = edge_index
    opt.data.edge_type = opt.data.edge_type[edge_mask]
    opt.data.train_idx = mapping[:opt.data.train_idx.size(0)]
    opt.data.test_idx = mapping[opt.data.train_idx.size(0):]
    opt.relations = opt.dataset.num_relations
    opt.num_classes = opt.dataset.num_classes
    opt.data = opt.dataset[0]
    opt.num_nodes = opt.data.num_nodes
    total_train_acc = list()
    total_test_acc = list()
    no_valid_info_format = 'Run: [{}]\t Epoch: [{}]\t Loss: {: .6f} \t Train Accuracy: {: .4f}'
    valid_info_format = 'Run: [{}]\t Epoch: [{}]\t Train Loss: {: .6f} \t Train Accuracy: {: .4f} \t' \
                        'Valid Loss: {: .4f} ' '\t Valid Accuracy: {: .4f}'

    opt.printer.info('=== Dataset info ===')
    opt.printer.info(f'Number of training nodes: {opt.data.train_idx.shape[0]} \t Number of testing nodes: '
                     f'{opt.data.test_idx.shape[0]}'
                     f'\t Total Number of labelled nodes: {opt.data.train_idx.shape[0] + opt.data.test_idx.shape[0]}')
    opt.printer.info(f'Number of classes in this graph: {opt.num_classes} \t Number of nodes: {opt.num_nodes} \t '
                     f'Number of relation types: {opt.relations}')

    opt.printer.info('==== Initializing the Optimizer ====')
    if opt.validation > 0:
        [train_idx, valid_idx, train_y, valid_y] = [i.to(opt.device)
                                                    for i in train_test_split(opt.data.train_idx, opt.data.train_y,
                                                                              test_size=opt.validation)]
        len_val = len(valid_idx)
    else:
        len_val = 0
        train_idx, train_y = opt.data.train_idx.to(opt.device), opt.data.train_y.to(opt.device)
    test_idx, test_y = opt.data.test_idx.to(opt.device), opt.data.test_y.to(opt.device)
    for run in range(1, opt.runs+1):
        data = opt.data.to(opt.device)
        criterion = F.nll_loss
        best_test = 0
        train_acc, test_acc = 0, 0
        model = DenseR(opt).to(opt.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2norm)

        for epoch in range(1, opt.total_epochs + 1):
            # Train Step
            train_loss = train()

            # Validation step
            if opt.validation > 0:
                valid_loss, train_acc, valid_acc = valid()
                opt.printer.info(valid_info_format.format(run, epoch, train_loss, train_acc, valid_loss, valid_acc))
            else:
                train_acc, test_acc = test()
                opt.printer.info(no_valid_info_format.format(run, epoch, train_loss, train_acc))

            # Test Step
            train_acc, test_acc = test()
        total_train_acc.append(train_acc)
        total_test_acc.append(test_acc)
        opt.printer.info(f'Test Accuracy in run {run} is: {test_acc}')
        opt.printer.info(f'Total Train Accuracy after run {run} is: {s.mean(total_train_acc):.4f}')
        opt.printer.info(f'Total Test Accuracy after run {run} is: {s.mean(total_test_acc):.4f}')
        opt.printer.info(f'Best Test Accuracy in {run} runs is: {max(total_test_acc):.4f}')
