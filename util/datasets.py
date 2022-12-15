import csv
import os
import PIL
import cv2
import numpy as np
import torch
import glob
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
from PIL import Image
import tqdm
import SimpleITK as sitk
import json
import time
from joblib import Parallel, delayed


class VinBigData(Dataset):
    def __init__(self, main_path, img_size=224, transform=None, mode="train", extra_data=0, ar=0.):
        super(VinBigData, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        if mode == "train":
            train_normal = pd.read_csv(os.path.join(self.root, "train_normal.csv"))  # 4000
            for img_name in tqdm(train_normal["image_id"]):
                # dicom = pydicom.read_file(os.path.join(self.root, "train_png_512", img_name+".png"))
                # img = dicom.pixel_array
                # if dicom.PhotometricInterpretation == "MONOCHROME1":
                #     img = np.max(img) - img
                #
                # img = ((img - np.min(img)) * 1.0 / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
                img = Image.open(os.path.join(self.root, "train_png_512", img_name + ".png")).convert('L').resize(
                    (img_size, img_size), resample=Image.BILINEAR)
                # img = Image.fromarray(img).convert('L').resize((img_size, img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(0)

            if extra_data > 0:
                extra_csv = pd.read_csv(os.path.join(self.root, "extra.csv"))
                normal_l = list(extra_csv.loc[extra_csv["class_id"] == 0, "image_id"])
                anomaly_l = list(extra_csv.loc[extra_csv["class_id"] == 1, "image_id"])

                anomaly_num = int(extra_data * ar)
                normal_num = extra_data - anomaly_num

                for img_name in tqdm(normal_l[:normal_num]):
                    img = Image.open(os.path.join(self.root, "train_png_512", img_name + ".png")).convert('L').resize(
                        (img_size, img_size), resample=Image.BILINEAR)

                    self.slices.append(img)
                    self.labels.append(0)

                for img_name in tqdm(anomaly_l[:anomaly_num]):
                    img = Image.open(os.path.join(self.root, "train_png_512", img_name + ".png")).convert('L').resize(
                        (img_size, img_size), resample=Image.BILINEAR)

                    self.slices.append(img)
                    self.labels.append(1)

        else:
            data_csv = pd.read_csv(os.path.join(self.root, "test.csv"))

            test_images = data_csv["image_id"]
            test_labels = data_csv["class_id"]
            for i in tqdm(range(len(test_images))):
                img_name = test_images[i]
                label = test_labels[i]
                img = Image.open(os.path.join(self.root, "train_png_512", img_name + ".png")).convert("L").resize(
                    (img_size, img_size), resample=Image.BILINEAR)

                self.slices.append(img)
                self.labels.append(label)

        self.anomaly_mask = [0] * len(self.slices)

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        anomaly_mask = self.anomaly_mask[index]
        return img, label, anomaly_mask

    def __len__(self):
        return len(self.slices)


class CovidX(Dataset):

    def __init__(self, main_path, img_size=224, transform=None, mode="train"):
        super(CovidX, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.image_names = []
        self.transform = transform if transform is not None else lambda x: x
        self.img_size = img_size
        self.mode = mode
        if mode == "train":
            negative_train = pd.read_csv(os.path.join(self.root, 'negative_train.txt'), delimiter=' ', header=None)
            negative_train.columns = ["patient_id", "filename", "class", "source"]
            self.image_names = negative_train['filename']
            self.labels = negative_train['class']
        else:
            test = pd.read_csv(os.path.join(self.root, 'test.txt'), delimiter=' ', header=None)
            test.columns = ["patient_id", "filename", "class", "source"]
            self.image_names = test['filename']
            self.labels = test['class']

    def __getitem__(self, index):
        # if self.mode == "train":
        image = Image.open(os.path.join(self.root, self.mode, self.image_names[index])).convert("L").resize(
            (self.img_size, self.img_size), resample=Image.BILINEAR)
        image = np.array(image)
        image = self.transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)


class RSNA(Dataset):
    def __init__(self, main_path, mode="train", img_size=224, transform=None, extra_data=0, ar=0.):
        super(RSNA, self).__init__()
        # self.file_path = []
        self.labels = []
        self.anomaly_mask = []
        self.slices = []
        self.mode = mode
        self.transform = transform if transform is not None else lambda x: x
        self.file_names = []
        main_path = os.path.join(main_path, 'RSNA')
        self.main_path = main_path
        bbox_csv = pd.read_csv(os.path.join(main_path, "stage_2_train_labels.csv"))
        self.bbox_csv = bbox_csv
        for label in os.listdir(main_path + '/' + self.mode + "/"):
            if label not in ['0', '1']:
                continue

            if label == 0:
                for file_name in os.listdir(main_path + '/' + self.mode + "/" + label):
                    data = sitk.ReadImage(main_path + '/' + self.mode + "/" + label + '/' + file_name)
                    data = sitk.GetArrayFromImage(data).squeeze()
                    img = Image.fromarray(data).convert('L').resize((img_size, img_size), resample=Image.BILINEAR)
                    self.slices.append(img)
                    self.labels.append(int(label))
                    self.anomaly_mask.append(torch.zeros((img_size, img_size)).long())
            else:
                for file_name in os.listdir(main_path + '/' + self.mode + "/" + label):
                    data = sitk.ReadImage(main_path + '/' + self.mode + "/" + label + '/' + file_name)
                    data = sitk.GetArrayFromImage(data).squeeze()
                    img = Image.fromarray(data).convert('L').resize((img_size, img_size), resample=Image.BILINEAR)
                    self.slices.append(img)
                    self.labels.append(int(label))
                    bboxes = np.array(
                        bbox_csv[bbox_csv["patientId"] == file_name.split(".")[0]][["x", "y", "width", "height"]])
                    bboxes = np.round(bboxes * img_size * 1.0 / 1024.).astype(np.uint8)
                    mask = np.zeros((img_size, img_size))
                    # bboxes = bboxes.astype(np.uint8)
                    # mask = np.zeros((1024, 1024)).astype(np.uint8)
                    for x, y, w, h in bboxes:
                        mask[y:y + h, x:x + w] = 1
                    mask = torch.tensor(mask).long()
                    # print(torch.sum(mask))
                    self.anomaly_mask.append(mask)
                    self.file_names.append(file_name)

        if extra_data > 0:  # only for training A
            anomaly_num = int(extra_data * ar)
            normal_num = extra_data - anomaly_num
            normal_l = sorted(os.listdir(main_path + '/' + 'NORMAL'))
            anomaly_l = sorted(os.listdir(main_path + '/' + 'ABNORMAL'))

            for file_name in tqdm(normal_l[:normal_num]):
                data = sitk.ReadImage(main_path + '/' + 'NORMAL' + '/' + file_name)
                data = sitk.GetArrayFromImage(data).squeeze()
                img = Image.fromarray(data).convert('L').resize((img_size, img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(0)
                self.anomaly_mask.append(torch.zeros((img_size, img_size)).long())

            for file_name in tqdm(anomaly_l[:anomaly_num]):
                data = sitk.ReadImage(main_path + '/' + 'ABNORMAL' + '/' + file_name)
                data = sitk.GetArrayFromImage(data).squeeze()
                img = Image.fromarray(data).convert('L').resize((img_size, img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(1)
                self.anomaly_mask.append(torch.zeros((img_size, img_size)).long())

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        file_name = self.file_names[index]
        anomaly_mask = self.anomaly_mask[index]
        if self.mode == 'test':
            return img, label, file_name
        else:
            return img, label

    def __len__(self):
        return len(self.slices)

    def save_file_through_name(self, file_name):
        base_dir = self.main_path + '/' + self.mode + "/"
        for label in ['1']:
            if os.path.exists(base_dir + label + '/' + file_name):
                data = sitk.ReadImage(base_dir + label + '/' + file_name)
                data = sitk.GetArrayFromImage(data).squeeze().astype(np.uint8)
                img = Image.fromarray(data).convert('L').resize((224, 224), resample=Image.BILINEAR)
                bboxes = np.array(
                    self.bbox_csv[self.bbox_csv["patientId"] == file_name.split(".")[0]][["x", "y", "width", "height"]])
                bboxes = bboxes.astype(np.uint8)
                mask = np.zeros((1024, 1024)).astype(np.uint8)
                for x, y, w, h in bboxes:
                    mask[y:y + h, x:x + w] = 1

                final_data = data - np.multiply(mask, data)
                final_data = Image.fromarray(final_data).convert('L').resize((224, 224), resample=Image.BILINEAR)
                # mask = torch.tensor(mask).long()
                final_data.save(
                    '/home/caosiqi/projects/mae-main-rsna/second_stage/final_data/' + file_name.split('.')[0] + '.jpg')
                # img_mask = Image.fromarray(mask).convert('L').resize((224, 224), resample=Image.BILINEAR)
                # # mask = torch.tensor(mask).long()
                # img_mask.save('/home/caosiqi/projects/mae-main-rsna/second_stage/wrong_masks/' + file_name.split('.')[0] + '.jpg')
            else:
                print('no such file')

def parallel_load(img_dir, img_list, img_size, verbose=0):
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert("L").resize(
            (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)


class VBD(Dataset):
    def __init__(self, main_path, img_size=224, transform=None, mode="train", extra_data=0, ar=0.):
        super(VBD, self).__init__()
        assert mode in ["train", "test"]

        self.root = main_path
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x
        self.mode = mode
        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        if mode == "train":
            train_normal = data_dict["train"]["0"]

            normal_l = data_dict["train"]["unlabeled"]["0"]
            abnormal_l = data_dict["train"]["unlabeled"]["1"]
            if extra_data > 0:
                abnormal_num = int(extra_data * ar)
                normal_num = extra_data - abnormal_num
            else:
                abnormal_num = 0
                normal_num = 0

            train_l = train_normal + normal_l[:normal_num] + abnormal_l[:abnormal_num]
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "train_png_512"), train_l, img_size)
            self.labels += (len(train_normal) + normal_num) * [0] + abnormal_num * [1]
            self.img_id += [img_name.split('.')[0] for img_name in train_l]
            print("Loaded {} normal images, "
                  "{} (unlabeled) normal images, "
                  "{} (unlabeled) abnormal images. {:.3f}s".format(len(train_normal), normal_num, abnormal_num,
                                                                   time.time() - t0))

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "train_png_512"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]
        if self.mode == 'test':
            return img, label, img_id
        else:
            return img, label

    def __len__(self):
        return len(self.slices)


class pretrain(Dataset):
    def __init__(self, main_path, transform=None, anomaly_rate=0.):
        super(pretrain, self).__init__()
        self.transform = transform if transform is not None else lambda x: x
        self.anomaly_rate = anomaly_rate
        self.normal_vin_dirs = []
        self.normal_chest_dirs = []
        self.anomaly_vin_dirs = []
        self.anomaly_chest_dirs = []
        # rsna
        normal_rsna_dirs = os.listdir(os.path.join(main_path, 'RSNA', 'train', '0'))
        self.normal_rsna_dirs = [os.path.join(main_path, 'RSNA', 'train', '0', file_name) for file_name in
                                 normal_rsna_dirs]
        # vinbigdata
        vin_normal = pd.read_csv(os.path.join(main_path, 'VinBigData', 'normal_train.csv'))  # 4000
        vin_normal.columns = [0, 1, 2, 3, 4, 5, 6, 7]
        for img_name in vin_normal[0]:
            self.normal_vin_dirs.append(os.path.join(main_path, 'VinBigData', 'train_png_512', img_name + '.png'))
        vin_anomaly = pd.read_csv(os.path.join(main_path, 'VinBigData', 'anomaly_train.csv'))  # 4000
        vin_anomaly.columns = [0, 1, 2, 3, 4, 5, 6, 7]
        for img_name in vin_anomaly[0]:
            self.anomaly_vin_dirs.append(os.path.join(main_path, 'VinBigData', 'train_png_512', img_name + '.png'))
        # chestxray
        chest_normal = pd.read_csv(os.path.join(main_path, 'ChestX-ray8', 'normal_train.csv'))
        chest_normal.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for img_name in chest_normal[0]:
            self.normal_chest_dirs.append(os.path.join(main_path, 'ChestX-ray8', 'images', img_name))

        chest_anomaly = pd.read_csv(os.path.join(main_path, 'ChestX-ray8', 'anomaly_train.csv'))
        chest_anomaly.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for img_name in chest_anomaly[0]:
            self.anomaly_chest_dirs.append(os.path.join(main_path, 'ChestX-ray8', 'images', img_name))

        self.normal_dirs = self.normal_vin_dirs + self.normal_chest_dirs + self.normal_rsna_dirs  # 96028
        self.anomaly_dirs = self.anomaly_chest_dirs + self.anomaly_vin_dirs  # 87855
        np.random.seed(0)
        np.random.shuffle(self.normal_dirs)
        np.random.shuffle(self.anomaly_dirs)
        self.dirs = self.anomaly_dirs[:int(self.anomaly_rate * len(self.anomaly_dirs))] + \
                    self.normal_dirs[:int((1 - self.anomaly_rate) * len(self.anomaly_dirs))]

    def __getitem__(self, item):
        img_path = self.dirs[item]
        image = None
        if img_path.split('.')[-1] != 'png' or 'jpg':
            data = sitk.ReadImage(img_path)
            data = sitk.GetArrayFromImage(data).squeeze()
            image = Image.fromarray(data).convert('L').resize((224, 224), resample=Image.BILINEAR)
        else:
            image = Image.open(img_path).convert("L").resize((224, 224), resample=Image.BILINEAR)
            image = np.array(image)
            image = self.transform(image)
        image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.dirs)  # 96028


class mvtec(Dataset):

    def __init__(self, root_dir, class_name, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform if transform is not None else lambda x: x
        self.mode = mode
        self.root_dir = root_dir
        if mode == 'train':
            self.image_paths = sorted(glob.glob(root_dir+"/mvtec/{}/train/good/*.png".format(class_name)))
        else:
            self.image_paths = sorted(glob.glob(root_dir+"/mvtec/{}/test/*/*.png".format(class_name)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path = self.image_paths[idx]
            image = Image.open(img_path).resize((224, 224), resample=Image.BILINEAR)
            image = np.array(image)
            if image.shape == (224, 224):
                image = cv2.merge((image, image, image))
            # print(image.shape)
            image = self.transform(image)
            return image, 0
        else:
            img_path = self.image_paths[idx]
            print(img_path)
            image = Image.open(img_path).resize((224, 224), resample=Image.BILINEAR)
            image = np.array(image)
            if image.shape == (224, 224):
                image = cv2.merge((image, image, image))
            image = self.transform(image)
            label = 1
            if 'good' in img_path:
                label = 0
            else:
                label = 1
            print(label)
            return image, label, img_path
