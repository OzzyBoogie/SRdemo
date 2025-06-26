import glob
import random
import numpy as np
import h5py
import cv2
import torch
import torch.utils.data as udata
import os


def normalize(x):
    return x / 255.0


# def Im2Patch(img, win, stride=1):
#     k = 0
#     print("img.shape", img.shape)
#     endc = img.shape[0]
#     endw = img.shape[1]
#     endh = img.shape[2]
#     patch = img[:, 0 : endw - win + 0 + 1 : stride, 0 : endh - win + 0 + 1 : stride]
#     TotalPatNum = patch.shape[1] * patch.shape[2]
#     Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
#     for i in range(win):
#         for j in range(win):
#             patch = img[
#                 :, i : endw - win + i + 1 : stride, j : endh - win + j + 1 : stride
#             ]
#             Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
#             k = k + 1
#     return Y.reshape([endc, win, win, TotalPatNum])


def prepareH5(data_path):
    print("Preparing H5")
    # HIGH
    print("Searching in:", os.path.join(data_path, "HIGH", "*.png"))
    files = glob.glob(os.path.join(data_path, "HIGH", "*.png"))
    print("Found", len(files), "files")
    files.sort()
    h5f_train = h5py.File(data_path + "/HIGH.h5", "w")
    h5f_eval = h5py.File(data_path + "/HIGH_VAL.h5", "w")

    train_num = 0
    eval_num = 0

    for i in range(len(files)):
        img = cv2.imread(files[i])
        img = np.transpose(img, (2, 0, 1))
        img = np.float32(normalize(img))
        if i < len(files) * 0.9:
            h5f_train.create_dataset(str(train_num), data=img)
            train_num += 1
        else:
            h5f_eval.create_dataset(str(eval_num), data=img)
            eval_num += 1

    # LOW
    files = glob.glob(os.path.join(data_path, "LOW", "*.png"))
    files.sort()
    h5f_train = h5py.File(data_path + "/LOW.h5", "w")
    h5f_eval = h5py.File(data_path + "/LOW_VAL.h5", "w")

    train_num = 0
    eval_num = 0

    for i in range(len(files)):
        img = cv2.imread(files[i])
        img = np.transpose(img, (2, 0, 1))
        img = np.float32(normalize(img))
        if i < len(files) * 0.9:
            h5f_train.create_dataset(str(train_num), data=img)
            train_num += 1
        else:
            h5f_eval.create_dataset(str(eval_num), data=img)
            eval_num += 1
    h5f_train.close()
    h5f_eval.close()


class Dataset(udata.Dataset):
    def __init__(self, train=True, data_path="X2"):
        super().__init__()
        self.train = train
        self.data_path = data_path
        # self.h5f = h5py.File(data_path, "r")
        print(self.data_path)
        if self.train:
            self.hr_data = h5py.File(self.data_path + "/HIGH.h5", "r")
            self.lr_data = h5py.File(self.data_path + "/LOW.h5", "r")
        else:
            self.hr_data = h5py.File(self.data_path + "/HIGH_VAL.h5", "r")
            self.lr_data = h5py.File(self.data_path + "/LOW_VAL.h5", "r")

    def __len__(self):
        return len(self.hr_data)

    def __getitem__(self, index):
        hr = self.hr_data[str(index)][:]  # (C, H, W)
        lr = self.lr_data[str(index)][:]

        # print(index)
        # print(hr.shape)

        # 转为 C, H, W
        # hr = torch.from_numpy(hr.transpose((2, 0, 1))).float() / 255
        # lr = torch.from_numpy(lr.transpose((2, 0, 1))).float() / 255

        return lr, hr
