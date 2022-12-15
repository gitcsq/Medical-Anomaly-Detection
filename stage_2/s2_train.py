
import os
import torch
import numpy as np
import pandas as pd
import timm
import timm.optim.optim_factory as optim_factory
import argparse
import datetime
import json
import time
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
import sys

sys.path.append("..")
from stage_2.s2_models_mae import cls_token_model, fc_model
import stage_2.s2_dataset as s2_dataset
import stage_2.s2_test as s2_test
from stage_2.s2_engine_pretrain import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc


def main(train_dir_name, test_dir_name, args):
    misc.init_distributed_mode(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # 数据集类只是一个读取路径中文件的功能。只需要给路径就可以。生成路径需要根据args
    dataset_train = s2_dataset.latent_dataset(train_dir=train_dir_name, test_dir=test_dir_name, random_p=args.random_p)
    dataset_test = s2_dataset.latent_dataset(mode='test', train_dir=train_dir_name, test_dir=test_dir_name, random_p=args.random_p)
    print(dataset_test.__len__())
    # 周围-中间：周围patch数
    patch_num = dataset_train.num_patches
    print(patch_num)
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    # define the model
    in_chans = 3 if args.dataset_name=='mvtec' else 1
    # patches_num: cls_token_model不需要；而fc_model需要。就都写在这里了。
    # 模型对于数据集、source敏感。source为latent时，和什么数据集无关；为img时，模型前要添加patch_embed把图像块变成特征，
    # 而这个模块对于原始图片的通道敏感。
    # 为了统一以上三个参数对模型的影响。此模型在任何参数组合下，
    # 都包含了：patch_embed（仅当source=="img"时用）、需要输入patches_num
    model = eval(args.model_name)(in_chans=in_chans, num_patches=patch_num, depth=args.depth, k=args.mlp_k)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and ((epoch == 0 or epoch % args.save_freq == 0) or epoch + 1 == args.epochs):
            misc.save_model_2(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, depth=args.depth, win_size=args.win_size, mlp_k=args.mlp_k,
            model_name=args.model_name)
            print('testing...')
            auc, ap = s2_test.generate_test_label(data_loader_test, model, args.source)
            save_results(args, auc, ap, epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def save_results(args, auc, ap, epoch):
    if os.path.exists(args.output_dir + "result.xlsx"):
        df = pd.read_excel(args.output_dir + "result.xlsx")
        df_new = pd.DataFrame([(args.s1_chkpt, args.model_name, args.win_size, args.depth, args.mlp_k, epoch, auc, ap)],
                          columns=['s1_chkpt', 'model_name', 'win_size', 'depth', 'k', 'epoch', 'auc', 'ap'])
        df = pd.concat([df, df_new], ignore_index=True)
        print(df)
        df.to_excel(args.output_dir + "result.xlsx", index=False)
    else:
        df = pd.DataFrame([(args.s1_chkpt, args.model_name, args.win_size, args.depth, args.mlp_k, epoch, auc, ap)],
                          columns=['s1_chkpt', 'model_name', 'win_size', 'depth', 'k', 'epoch', 'auc', 'ap'])
        df.to_excel(args.output_dir + "result.xlsx", index=False)
