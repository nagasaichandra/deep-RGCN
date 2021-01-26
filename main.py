import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
# from sklearn.metrics import f1_score
from architecture import DeepArchitecture
from args import ArgsInit
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
    args = ArgsInit().initialize()
    args.printer.info(' === Downloading Dataset and Creating graphs ===')
    args.dataset = Entities('/data/entities/', args.dataset)
    node_idx = torch.cat([args.data.train_idx, args.data.test_idx], dim=0)
    node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, 2, args.data.edge_index, relabel_nodes=True)

    args.data.num_nodes = node_idx.size(0)
    args.data.edge_index = edge_index
    args.data.edge_type = args.data.edge_type[edge_mask]
    args.data.train_idx = mapping[:args.data.train_idx.size(0)]
    args.data.test_idx = mapping[args.data.train_idx.size(0):]
    args.relations = args.dataset.num_relations
    args.num_classes = args.dataset.num_classes
    args.data = args.dataset[0]
    args.num_nodes = args.data.num_nodes
    total_train_acc = list()
    total_test_acc = list()
    no_valid_info_format = 'Run: [{}]\t Epoch: [{}]\t Loss: {: .6f} \t Train Accuracy: {: .4f}'
    valid_info_format = 'Run: [{}]\t Epoch: [{}]\t Train Loss: {: .6f} \t Train Accuracy: {: .4f} \t' \
                        'Valid Loss: {: .4f} ' '\t Valid Accuracy: {: .4f}'
    args.printer.info('==== Initializing the Optimizer ====')
    if args.validation > 0:
        [train_idx, valid_idx, train_y, valid_y] = [i.to(args.device)
                                                    for i in train_test_split(args.data.train_idx, args.data.test_idx,
                                                                              test_size=args.validation)]
        len_val = len(valid_idx)
    else:
        len_val = 0
        train_idx, train_y = args.data.train_idx.to(args.device), args.data.train_y.to(args.device)
    test_idx, test_y = args.data.test_idx.to(args.device), args.data.test_y.to(args.device)
    for run in args.runs:
        data = args.data.to(args.device)
        criterion = F.nll_loss
        best_test = 0
        train_acc, test_acc = 0, 0
        model = DeepArchitecture(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

        for epoch in range(1, args.total_epochs + 1):
            # Train Step
            train_loss = train()

            # Validation step
            if args.validation > 0:
                valid_loss, train_acc, valid_acc = valid()
                args.printer.info(valid_info_format.format(run, epoch, train_loss, train_acc, valid_loss, valid_acc))
            else:
                train_acc, test_acc = test()
                args.printer.info(no_valid_info_format.format(run, epoch, train_loss, train_acc))

            # Test Step
            train_acc, test_acc = test()
        total_train_acc.append(train_acc)
        total_test_acc.append(test_acc)
        args.printer.info(f'Test Accuracy in run {run} is: {test_acc}')
        args.printer.info(f'Total Train Accuracy after run {run} is: {s.mean(total_train_acc):.4f}')
        args.printer.info(f'Total Test Accuracy after run {run} is: {s.mean(total_test_acc):.4f}')
        args.printer.info(f'Best Test Accuracy in {run} runs is: {max(total_test_acc):.4f}')
