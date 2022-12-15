import sys
sys.path.append("..")
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from stage_1 import models_mae
import torch
from util.datasets import *
from patchify import patchify, unpatchify


class FeatureProcessing(Dataset):
    def __init__(self, win_size=9, mode="train", type_idx=None):
        super(FeatureProcessing, self).__init__()
        types = ['none', 'dilated', 'dilated_padding_zero', 'dilated_padding_rotate_8',
                 'dilated_padding_rotate_4', 'dilated_padding_rotate_2']
        self.mode = mode
        one_image = (np.array(range(196)) + 1).reshape((14, 14))
        k = -np.ones((win_size, win_size))
        # 改patch_position_matrix。1~196代表了196个patch。在zero_padding中，197代表全0特征。
        ## 无padding
        ## 循环padding：把9个贴起来
        three_images = np.vstack((one_image, one_image, one_image))
        padding = np.hstack((three_images, three_images, three_images))
        self.patch_position_matrix = padding
        # 更改k，也就是卷积核。根据膨胀率来改。为1是要的patch，为-1是不要的
        ## 外面一圈是1
        if type_idx in [0, 4]:
            k = np.ones((win_size, win_size))
            k[1:-1, 1:-1] = -1
            self.win_kernel = k
        ## 8个点是1
        if type_idx in [1]:
            k[0, 0] = 1
            k[0, win_size - 1] = 1
            k[win_size - 1, 0] = 1
            k[win_size - 1, win_size - 1] = 1
            k[0, int((win_size - 1) / 2)] = 1
            k[int((win_size - 1) / 2), win_size - 1] = 1
            k[win_size - 1, int((win_size - 1) / 2)] = 1
            k[int((win_size - 1) / 2), 0] = 1
            self.win_kernel = k
        ## 4个点是1
        if type_idx in [2]:
            k[0, int((win_size - 1) / 2)] = 1
            k[int((win_size - 1) / 2), win_size - 1] = 1
            k[win_size - 1, int((win_size - 1) / 2)] = 1
            k[int((win_size - 1) / 2), 0] = 1
            self.win_kernel = k
        ## 2个点是1
        if type_idx in [3]:
            k[0, int((win_size - 1) / 2)] = 1
            k[win_size - 1, int((win_size - 1) / 2)] = 1
            self.win_kernel = k
        ## 指定百分比为1，随机采样：一圈的标号随机去百分比

        self.groups, self.centers = self.padding_convolution(self.win_kernel, self.patch_position_matrix)
        # if type_idx in [4]:
        #     full_index = np.random.permutation(self.groups.shape[1])[0: int(percentage*self.groups.shape[1])]
        #     new_groups = self.groups[:, full_index]
        #     self.groups = new_groups

        self.groups_flatten = self.groups.flatten()
        self.num_patches = self.groups.shape[1]
        print("num_patches:{}".format(self.num_patches))
        self.num_groups = (15 - win_size) ** 2 if type_idx in [0, 1] else 196

    def padding_convolution(self, k, data):
        win_size = k.shape[0]
        groups = []
        labels = []
        for i in range(14):
            for j in range(14):
                a = data[int(14 - (win_size - 1) / 2 + i): int(14 + (win_size - 1) / 2 + i + 1),
                    int(14 - (win_size - 1) / 2 + j)
                    : int(14 + (win_size - 1) / 2 + j + 1)]
                one_group_idx = np.multiply(k, a).flatten()
                one_group_idx = one_group_idx[np.where(one_group_idx >= 0)]
                one_group_idx = one_group_idx.astype(int)
                groups.append(one_group_idx)
                labels.append(data[14 + i, 14 + j])
        return np.array(groups) - 1, np.array(labels) - 1

    def convolution(self, k, data):
        n, m = data.shape
        win_size = k.shape[0]
        groups = []
        labels = []
        for i in range(n - win_size + 1):
            for j in range(m - win_size + 1):
                a = data[i:i + win_size, j:j + win_size]
                one_group_idx = np.multiply(k, a).flatten()
                one_group_idx = one_group_idx[np.where(one_group_idx >= 0)]
                one_group_idx = one_group_idx.astype(int)
                groups.append(one_group_idx)
                labels.append(data[int(i + (win_size - 1) / 2), int(j + (win_size - 1) / 2)])
        return np.array(groups) - 1, np.array(labels) - 1

    def getitem(self, orig, pos_embed):
        train_latent = orig.squeeze()
        print(self.groups_flatten)
        data_one_img = train_latent[self.groups_flatten]
        data_one_img = data_one_img.reshape((self.groups.shape[0], self.groups.shape[1], -1))  # k=13: (4, 48, 1024)
        label_one_img = train_latent[self.centers].unsqueeze(1)  # (4, 1, 1024)
        label_pos = pos_embed[self.centers].unsqueeze(1)
        return data_one_img, label_one_img, label_pos

    def getitem_img(self, patches, pos_embed):
        # patches: 196, 3, 16, 16
        data_one_img = patches[self.groups_flatten]  # ?, 3, 16, 16
        data_one_img = data_one_img.reshape((self.groups.shape[0], self.groups.shape[1], data_one_img.shape[1], 16, 16))
        # k=13: (4, 48, 3, 16, 16)
        label_one_img = patches[self.centers].unsqueeze(1)  # (4, 1, 3, 16, 16)
        data_pos = pos_embed[self.groups_flatten]
        data_pos = data_pos.reshape((self.groups.shape[0], self.groups.shape[1], -1))  # 4, 48, 1024
        label_pos = pos_embed[self.centers].unsqueeze(1)
        return data_one_img, label_one_img, data_pos, label_pos


def prepare_model(dataset_name, chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    if dataset_name == 'mvtec':
        in_chans = 3
    else:
        in_chans = 1
    model = getattr(models_mae, arch)(in_chans=in_chans)
    # load model
    checkpoint = torch.load(chkpt_dir)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def generate_latent(device, args, train_dir_name, test_dir_name):
    if args.dataset_name == 'mvtec':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        data_path = '/home/caiyu/datasets'
        dataset_train = eval(args.dataset_name)(data_path, class_name=args.class_name, transform=transform_train)
        dataset_test = eval(args.dataset_name)(data_path, mode='test', class_name=args.class_name,
                                               transform=transform_train)
    elif args.dataset_name == 'VBD':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        data_path = '/data/caiyu/datasets/Med-AD/VinCXR'
        dataset_train = eval(args.dataset_name)(data_path, transform=transform_train)
        dataset_test = eval(args.dataset_name)(data_path, mode='test', transform=transform_train)

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        data_path = '/data/caiyu'
        dataset_train = eval(args.dataset_name)(data_path, transform=transform_train)
        dataset_test = eval(args.dataset_name)(data_path, mode='test', transform=transform_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    win_size = args.win_size
    types = ['none', 'dilated', 'dilated_padding_zero', 'dilated_padding_rotate_8',
             'dilated_padding_rotate_4', 'dilated_padding_rotate_2']

    # s1 model
    model = prepare_model(args.dataset_name, args.s1_chkpt, 'mae_vit_large_patch16')
    model.to(device)
    feature_processing = FeatureProcessing(win_size, type_idx=args.type_idx)
    # 直接用照片预测
    if args.source == 'img':
        for batch_idx, data_train in enumerate(data_loader_train):
            img_train, label_train = data_train  # 1, 3, 224, 224
            channel_num = img_train.shape[1]
            pos_embed = model.pos_embed[0, 1:, :]  # 196, 1024
            img_train = img_train.squeeze(0)  # 3, 224, 224
            patches = patchify(np.array(img_train), (channel_num, 16, 16), 16)
            patches = patches.reshape((-1, channel_num, 16, 16))  # 196, 3, 16, 16
            patches = torch.FloatTensor(patches)
            pos_embed = pos_embed.detach().cpu()  # 196, 1024
            if args.type_idx == 2:
                patches = torch.cat([patches, torch.zeros_like(patches[:1])])
                pos_embed = torch.cat([pos_embed, torch.zeros_like(pos_embed[:1])])
            print(patches.shape, pos_embed.shape)
            data_one_img, label_one_img, data_pos, label_pos = feature_processing.getitem_img(patches, pos_embed)

            for data_group_idx in range(feature_processing.num_groups):
                np.savez(
                    train_dir_name + "/{}.npz".format(batch_idx * feature_processing.num_groups + data_group_idx),
                    data=data_one_img[data_group_idx], label=label_one_img[data_group_idx],
                    data_pos=data_pos[data_group_idx], label_pos=label_pos[data_group_idx])

                print(batch_idx * feature_processing.num_groups + data_group_idx)
                print('data:{}'.format(data_one_img[data_group_idx].shape))
                print('label:{}'.format(label_one_img[data_group_idx].shape))
                print('data_pos:{}'.format(data_pos[data_group_idx].shape))
                print('label_pos:{}'.format(label_pos[data_group_idx].shape))

        for batch_idx, data_test in enumerate(data_loader_test):
            img_test, label_test, file_name = data_test  # 1, 1, 224, 224
            channel_num = img_test.shape[1]
            pos_embed = model.pos_embed[0, 1:, :]  # 196, 1024
            img_test = img_test.squeeze(0)  # 3, 224, 224
            patches = patchify(np.array(img_test), (channel_num, 16, 16), 16)
            patches = patches.reshape((-1, channel_num, 16, 16))  # 196, 3, 16, 16
            patches = torch.FloatTensor(patches)
            pos_embed = pos_embed.detach().cpu()  # 196, 1024
            if args.type_idx == 2:
                patches = torch.cat([patches, torch.zeros_like(patches[:1])])
                pos_embed = torch.cat([pos_embed, torch.zeros_like(pos_embed[:1])])
            print(patches.shape, pos_embed.shape)
            data_one_img, label_one_img, data_pos, label_pos = feature_processing.getitem_img(patches, pos_embed)

            np.savez(test_dir_name + "/{}.npz".format(batch_idx),
                     data=data_one_img, truth=label_one_img, file_name=file_name,
                     label=label_test, data_pos=data_pos, label_pos=label_pos)
            print(batch_idx)
    # 用隐藏特征预测
    else:
        for batch_idx, data_train in enumerate(data_loader_train):
            img_train, label_train = data_train
            img_train = img_train.to(device)
            latent_train, pos_embed = model.generate_latent_for_2(img_train)  # 196, 1024
            enc = latent_train.detach().cpu().squeeze()  # 196, 1024
            pos_embed = pos_embed.detach().cpu().squeeze()  # 196, 1024
            # if args.type_idx == 2:
            #     enc = torch.cat([enc, torch.zeros_like(enc[:1])])
            #     pos_embed = torch.cat([pos_embed, torch.zeros_like(pos_embed[:1])])
            #     print(enc.shape)
            data_one_img, label_one_img, label_pos = feature_processing.getitem(enc, pos_embed)
            print(data_one_img.shape)
            for data_group_idx in range(feature_processing.num_groups):
                np.savez(train_dir_name + "/{}.npz".format(
                    batch_idx * feature_processing.num_groups + data_group_idx),
                         data=data_one_img[data_group_idx], label=label_one_img[data_group_idx],
                         data_pos=0, label_pos=label_pos[data_group_idx])
                print(batch_idx * feature_processing.num_groups + data_group_idx)
                print('data:{}'.format(data_one_img[data_group_idx].shape))
                print('label:{}'.format(label_one_img[data_group_idx].shape))
                print('label_pos:{}'.format(label_pos[data_group_idx].shape))

        for batch_idx, data_test in enumerate(data_loader_test):
            img_test, label_test, file_name = data_test
            img_test = img_test.to(device)
            latent_test, pos_embed = model.generate_latent_for_2(img_test)
            enc = latent_test.detach().cpu()
            pos_embed = pos_embed.detach().cpu()  # 196, 1024

            data_one_img, label_one_img, label_pos = feature_processing.getitem(enc, pos_embed)
            np.savez(test_dir_name + "/{}.npz".format(batch_idx),
                     data=data_one_img, truth=label_one_img, label=label_test, file_name=file_name,
                     data_pos=0, label_pos=label_pos)
            print(batch_idx)
