import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config


class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Implementation of Modeling graphs using Deep RGCNs')

        # base
        parser.add_argument('--phase', default='train', type=str, help='train(default) or test')
        parser.add_argument('--use_cpu', action='store_true', help='if this arg is given, then cpu is used')
        parser.add_argument('--runs', default=10, type=int, help='the number of times to run the model for given '
                                                                 'epochs')
        parser.add_argument('--validation', default='0.0', type=float, help='if greater than 0, that percent of '
                                                                            'training set is used as validation set. ')
        # dataset
        parser.add_argument('--data_dir', type=str, default='/data/entity')
        parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size (default:8)')
        parser.add_argument('--dataset', type=str, default='MUTAG', help='dataset to train and test on: could be '
                                                                         'MUTAG(default), AIFB, BGS, AM (All rdf format'
                                                                         'graphs under entities)')
        # basic train args
        parser.add_argument('--total_epochs', default=50, type=int, help='number of total iterations to run')
        parser.add_argument('--save_freq', default=5, type=int, help='save model per num of epochs')
        parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate set')
        parser.add_argument('--multi-gpus', action='store_true', help='use multi-gpus?')
        parser.add_argument('--postname', default='', type=str, help='postname of saved file')
        parser.add_argument('--l2norm', default='0.0005', type=float, help='weight decay for optimizer')
        # model args
        parser.add_argument('--kernel_size', default=20, type=int, help='max neighbor num with each kernel (default:20)'
                            )
        parser.add_argument('--block_type', default='dense', type=str, help='graph backbone (could be plain, dense or '
                                                                          'res (default)')
        parser.add_argument('--n_layers', default=7, type=int, help='number of layers in (GCN) backbone block')
        parser.add_argument('--act', default='relu', type=str, help='activation layer: could be prelu, leakyrelu, '
                                                                    'relu(default)')
        parser.add_argument('--norm', default='layer', type=str, help='normalization type: could be batch (default), '
                                                                      'layer, or instance normalization')
        parser.add_argument('--bias', default=True, type=bool, help='bias of conv layers (true(default) or false)')
        parser.add_argument('--dropout', default=0.2, type=float, help='dropout ratio (for MLP layers) (default 0.2)')
        parser.add_argument('--channels', default=16, type=int, help='number of channels of deep features')
        parser.add_argument('--dec', default='basis', type=str, help='Type of decomposition used in R-GCN layers')
        parser.add_argument('--bases', default=30, type=int, help='Number of bases supported for basis decomposition')
        parser.add_argument('--aggr', default='mean', type=str, help='type of aggregator to be used as R-GCN feature '
                                                                     'aggregator, could be "mean" (default), "max", '
                                                                     '"sum"')
        # saving checkpoints
        parser.add_argument('--ckpt_path', type=str, default='')
        parser.add_argument('--save_best_only', default=True, type=bool, help='only save best model')

        args = parser.parse_args()

        dir_path = os.path.dirname(os.path.abspath(__file__))
        args.task = os.path.basename(dir_path)
        args.post = '-'.join([args.task, args.dataset, args.block_type, str(args.n_layers), str(args.channels)])
        if args.postname:
            args.post += '-' + args.postname
        args.time = datetime.datetime.now().strftime('%y%m%d')

        if not args.ckpt_path:
            args.save_path = os.path.join(dir_path, 'checkpoints/ckpts'+'-'+ args.post + '-' + args.time)
        else:
            args.save_path = os.path.join(args.ckpt_path, 'checkpoints/ckpts' + '-' + args.post + '-' + args.time)

        args.logdir = os.path.join(dir_path, 'logs/' + args.post + '-'+args.time)

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args

    def make_dir(self):
        if not os.path.exists(self.args.logdir):
            os.makedirs(self.args.logdir)
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def print_args(self):
        # self.args.printer args
        self.args.printer.info("==========       System Configuration      =============")
        for arg, content in self.args.__dict__.items():
            self.args.printer.info("{}:{}".format(arg, content))
        self.args.printer.info("==========     CONFIG END    =============")
        self.args.printer.info("\n")
        self.args.printer.info('===> Phase is {}.'.format(self.args.phase))

    def logging_init(self):
        if not os.path.exists(self.args.logdir):
            os.makedirs(self.args.logdir)
        ERROR_FORMAT = "%(message)s"
        DEBUG_FORMAT = "%(message)s"
        LOG_CONFIG = {'version': 1,
                      'formatters': {'error': {'format': ERROR_FORMAT},
                                     'debug': {'format': DEBUG_FORMAT}},
                      'handlers': {'console': {'class': 'logging.StreamHandler',
                                               'formatter': 'debug',
                                               'level': logging.DEBUG},
                                   'file': {'class': 'logging.FileHandler',
                                            'filename': os.path.join(self.args.logdir, self.args.post + '.log'),
                                            'formatter': 'debug',
                                            'level': logging.DEBUG}},
                      'root': {'handlers': ('console', 'file'), 'level': 'DEBUG'}
                      }
        logging.config.dictConfig(LOG_CONFIG)
        self.args.printer = logging.getLogger(__name__)

    def initialize(self):
        if self.args.phase == 'train':
            self.args.epoch = -1
            self.make_dir()

        self.set_seed(812)
        self.logging_init()
        self.print_args()
        return self.args