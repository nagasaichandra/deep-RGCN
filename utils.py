from torch import nn
# from torch.nn import Sequential as Seq, Linear as Lin
# import os
# import torch
# import shutil
# from collections import OrderedDict
# import logging
# import numpy as np


def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act_type.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm_type, nc):
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


# def save_ckpt(model, optimizer, loss, epoch, save_path, name_pre, name_post='best'):
#     model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
#     state = {
#             'epoch': epoch,
#             'model_state_dict': model_cpu,
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss
#         }
#
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#         print("Directory ", save_path, " is created.")
#
#     filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
#     torch.save(state, filename)
#     print('model has been saved as {}'.format(filename))


# def save_checkpoint(state, is_best, save_path, postname):
#     filename = '{}/{}_{}.pth'.format(save_path, postname, int(state['epoch']))
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, '{}/{}_best.pth'.format(save_path, postname))
#
#
# def change_ckpt_dict(model, optimizer, scheduler, util):
#
#     for _ in range(util.epoch):
#         scheduler.step()
#     is_best = (util.test_value < util.best_value)
#     util.best_value = min(util.test_value, util.best_value)
#
#     model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
#     # optim_cpu = {k: v.cpu() for k, v in optimizer.state_dict().items()}
#     save_checkpoint({
#         'epoch': util.epoch,
#         'state_dict': model_cpu,
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'best_value': util.best_value,
#     }, is_best, util.save_path, util.post)