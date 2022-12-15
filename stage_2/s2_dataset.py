import os

import numpy as np
import torch
from torch.utils.data import Dataset


class latent_dataset(Dataset):
    def __init__(self, train_dir, test_dir, random_p, mode="train"):
        super(latent_dataset, self).__init__()
        self.mode = mode
        self.train_dir = train_dir
        self.test_dir = test_dir
        data = torch.Tensor(np.load(train_dir + "0.npz")['data'])
        self.num_patches = data.shape[0]
        self.random_p = random_p
        all_tensors_test = np.load(self.test_dir + "{}.npz".format(str(0)))
        test_latent = torch.Tensor(all_tensors_test['data'])
        print(test_latent.shape)

    def __getitem__(self, index):
        index_random = np.random.permutation(self.num_patches)[0:int(self.random_p*self.num_patches)]
        # print('random_id every couple{}'.format(index_random))
        if self.mode == 'train':
            all_tensors = np.load(self.train_dir + "{}.npz".format(str(index)))
            data = torch.Tensor(all_tensors['data'])[index_random]
            label = torch.Tensor(all_tensors['label'])
            data_pos = torch.Tensor(all_tensors['data_pos'])
            # print(data.shape, data_pos.shape)
            label_pos = torch.Tensor(all_tensors['label_pos'])
            return data, label, data_pos, label_pos
        else:
            all_tensors_test = np.load(self.test_dir + "{}.npz".format(str(index)))
            test_latent = torch.Tensor(all_tensors_test['data'])[:, index_random, :]
            test_truth = torch.Tensor(all_tensors_test['truth'])
            test_label = torch.IntTensor(all_tensors_test['label']).squeeze()
            test_label = int(test_label.numpy())
            data_pos = torch.Tensor(0)
            label_pos = torch.Tensor(all_tensors_test['label_pos'])

            file_name = str(all_tensors_test['file_name'])
            return test_latent, test_truth, data_pos, label_pos, test_label, file_name

    def __len__(self):
        if self.mode == 'train':
            file_num = len(os.listdir(self.train_dir))
            return file_num
        else:
            file_num = len(os.listdir(self.test_dir))
            return file_num

