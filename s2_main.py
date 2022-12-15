import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
from util.datasets import *
import tensorflow as tf
from stage_1 import models_mae
import shutil
from sklearn.metrics import roc_auc_score, average_precision_score

import argparse
import datetime
import json
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import timm
from stage_2.s2_train import main as s2_train_main
from stage_2.s2_test import main as s2_test_main
from stage_2.generate import generate_latent


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')


    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./ft-9',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./ft-9',
                        help='path where to tensorboard log')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # distributed training parameters
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # 可选参数
    ## 产生第一阶段隐藏特征
    parser.add_argument('--s1_chkpt',
                        default='/home/caosiqi/projects/mae-main-rsna/output_dir/RSNA/ep999.pth', type=str)
    parser.add_argument('--win_size', default=9, type=int)
    parser.add_argument('--type_idx', default=0, type=int,
                        help='none''dilated''dilated_padding_zero''dilated_padding_rotate_8'
                             'dilated_padding_rotate_4''dilated_padding_rotate_2')
    parser.add_argument('--always_generate', default=False, type=bool)
    parser.add_argument('--dataset_name', default='RSNA', type=str, help='RSNA, CovidX, pretrain, mvtec')
    parser.add_argument('--class_name', default='bottle', type=str)
    parser.add_argument('--source', default='latent', type=str)

    ## 第二阶段模型
    parser.add_argument('--model_name', default='cls_token_model', type=str,
                        help='choose from {cls_token_model, fc_model}')
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--mlp_k', default=4, type=int)

    ## 第二阶段训练参数
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--mode', default='train_eval', type=str, help='train_eval or test')
    parser.add_argument('--save_freq', default=20, type=int)

    ## 第二阶段测试参数
    parser.add_argument('--s2_chkpt', type=str)  # 可通过文件路径得到第二阶段模型参数、第二阶段数据参数（第一阶段隐藏特征参数）
    return parser


def refine_args(args):
    """
    生成train_dir_name和test_dir_name, 修改根据一部分必须输入的参数修改args另一部分参数（模式train_eval主要是output_dir,
    模式test有更多参数，见代码）
    """
    if args.mode == 'train_eval':
        types = ['none', 'dilated', 'dilated_padding_zero', 'dilated_padding_rotate_8',
                 'dilated_padding_rotate_4', 'dilated_padding_rotate_2']
        s1_chkpt_ = args.s1_chkpt.replace('/', '-').replace('.', ',').split('-ep')[-1].split(',pth')[0]
        if args.dataset_name == 'mvtec':
            train_dir_name = "./stage_2/s2_data/mvtec/{}/{}/{}/train_{}_{}/" \
                .format(args.class_name, s1_chkpt_, types[args.type_idx], args.source, str(args.win_size))
            test_dir_name = "./stage_2/s2_data/mvtec/{}/{}/{}/test_{}_{}/" \
                .format(args.class_name, s1_chkpt_, types[args.type_idx], args.source, str(args.win_size))

        else:
            train_dir_name = "./stage_2/s2_data/{}/{}/{}/train_{}_{}/" \
                .format(args.dataset_name, s1_chkpt_, types[args.type_idx], args.source, str(args.win_size))
            test_dir_name = "./stage_2/s2_data/{}/{}/{}/test_{}_{}/" \
                .format(args.dataset_name, s1_chkpt_, types[args.type_idx], args.source, str(args.win_size))

        os.makedirs(train_dir_name, exist_ok=True)
        os.makedirs(test_dir_name, exist_ok=True)

        types = ['none', 'dilated', 'dilated_padding_zero', 'dilated_padding_rotate_8',
                 'dilated_padding_rotate_4', 'dilated_padding_rotate_2']
        if args.dataset_name == 'mvtec':
            args.output_dir = "./output_dir/mvtec/{}/{}/{}/{}/" \
                .format(args.class_name, s1_chkpt_, types[args.type_idx], args.source)
        else:
            args.output_dir = "./output_dir/{}/{}/{}/{}/" \
                .format(args.dataset_name, s1_chkpt_, types[args.type_idx], args.source)
        args.log_dir = args.output_dir
    else:
        s2_chkpt = args.s2_chkpt
        args.output_dir = os.path.dirname(args.s2_chkpt) + '/'
        train_dir_name = None
        # /home/caosiqi/projects/mae-main-rsna/output_dir/RSNA/.-output_dir.../none/latent/cls_token_model-en4-mlp4-ep0-win9.pth
        args.win_size = int(s2_chkpt.split('.')[0].split('-')[-1][3:])
        args.epoch = int(s2_chkpt.split('.')[0].split('-')[-2][2:])
        args.mlp_k = int(s2_chkpt.split('.')[0].split('-')[-3][3:])
        args.depth = int(s2_chkpt.split('.')[0].split('-')[-4][2:])
        args.model_name = s2_chkpt.split('.')[0].split('-')[-5].split('/')[-1]
        type = s2_chkpt.split('/')[-3]
        args.source = s2_chkpt.split('/')[-2]
        s1_chkpt_ = s2_chkpt.split('/')[-4]
        args.s1_chkpt = s1_chkpt_.replace('-', '/').replace(',', '.')

        if s2_chkpt.split('/')[-6] == 'mvtec':
            class_name = s2_chkpt.split('/')[-5]
            args.dataset_name = s2_chkpt.split('/')[-6]
            train_dir_name = "./stage_2/s2_data/mvtec/{}/{}/{}/train_{}_{}/" \
                .format(class_name, s1_chkpt_, type, args.source, args.win_size)
            test_dir_name = "./stage_2/s2_data/mvtec/{}/{}/{}/test_{}_{}/" \
                .format(class_name, s1_chkpt_, type, args.source, args.win_size)
        else:
            args.dataset_name = s2_chkpt.split('/')[-5]
            train_dir_name = "./stage_2/s2_data/{}/{}/{}/train_{}_{}/" \
                .format(args.dataset_name, s1_chkpt_, type, args.source, args.win_size)
            test_dir_name = "./stage_2/s2_data/{}/{}/{}/test_{}_{}/" \
                .format(args.dataset_name, s1_chkpt_, type, args.source, args.win_size)

    return train_dir_name, test_dir_name, args


def main():

    args = get_args_parser()
    args = args.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(args.device)
    assert args.mode in ['train_eval', 'test']
    # train_dir_name和test_dir_name：s2_dataset要从这个文件夹提取数据
    train_dir_name, test_dir_name, args = refine_args(args)
    print(train_dir_name)
    #  不论目标路径有无数据，都重新生成（这里是清空该路径数据）
    if args.always_generate:
        shutil.rmtree(train_dir_name)
        shutil.rmtree(test_dir_name)
        os.makedirs(train_dir_name, exist_ok=True)
        os.makedirs(test_dir_name, exist_ok=True)
    # 如果该路径无数据，根据args参数生成数据到train_dir_name和test_dir_name这两个路径
    if os.listdir(train_dir_name) == []:
        print('generating s1 {}...'.format(args.source))
        generate_latent(device, args, train_dir_name, test_dir_name)
    # 如果训练，则去s2_train这个文件主函数，反之去s2_test这个文件主函数
    if args.mode == 'train_eval':
        s2_train_main(train_dir_name, test_dir_name, args)
    else:
        s2_test_main(train_dir_name, test_dir_name, args)


if __name__ == '__main__':
    main()
