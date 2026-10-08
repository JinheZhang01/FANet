import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import gaussian_filter


class TOFDSR_Dataset(Dataset):

    def __init__(self, root_dir="./dataset/", scale=4, downsample='real', train=False, txt_file='./TOFDSR_Test.txt',
                 transform=None, isNoisy=False, blur_sigma=1.2, patch_size=256):

        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.downsample = downsample
        self.train = train
        self.isNoisy = isNoisy
        self.blur_sigma = blur_sigma
        self.image_list = txt_file
        self.patch_size = patch_size

        with open(self.image_list, 'r') as f:
            self.filename = f.readlines()

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):

        sample_path = self.filename[idx].strip('\n')
        sample_path_ = sample_path.split(',')
        rgb_path = sample_path_[0]
        gt_path = sample_path_[1]
        lr_path = sample_path_[2]
        name = gt_path[20:-4]

        rgb_path = os.path.join(self.root_dir, rgb_path)
        gt_path = os.path.join(self.root_dir, gt_path)
        lr_path = os.path.join(self.root_dir, lr_path)


        image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)

        if self.downsample == 'real':
            gt = np.array(Image.open(gt_path)).astype(np.float32)
            h, w = gt.shape
            lr = np.array(Image.open(lr_path).resize((w // self.scale, h // self.scale), Image.BICUBIC)).astype(
                np.float32)

        else:
            gt_img = Image.open(gt_path)
            w, h = gt_img.size
            lr = np.array(gt_img.resize((w // self.scale, h // self.scale), Image.BICUBIC)).astype(np.float32)
            gt = np.array(gt_img).astype(np.float32)


        if self.train:

            if h > self.patch_size and w > self.patch_size:

                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)


                image = image[y: y + self.patch_size, x: x + self.patch_size, :]
                gt = gt[y: y + self.patch_size, x: x + self.patch_size]


                x_lr = x // self.scale
                y_lr = y // self.scale
                p_lr = self.patch_size // self.scale
                lr = lr[y_lr: y_lr + p_lr, x_lr: x_lr + p_lr]
        # ==========================================

        image_max = np.max(image)
        image_min = np.min(image)

        if image_max - image_min != 0:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = image - image_min

        if self.train:
            max_out = 5000.0
            min_out = 0.0
            lr = (lr - min_out) / (max_out - min_out)
            gt = (gt - min_out) / (max_out - min_out)
        else:
            max_out = 5000.0
            min_out = 0.0
            lr = (lr - min_out) / (max_out - min_out)

        lr_minn = np.min(lr)
        lr_maxx = np.max(lr)

        if not self.train:
            np.random.seed(42)

        if self.isNoisy:
            lr = gaussian_filter(lr, sigma=self.blur_sigma)
            gaussian_noise = np.random.normal(0, 0.07, lr.shape)
            lr = lr + gaussian_noise
            lr = np.clip(lr, lr_minn, lr_maxx)

        if self.transform:
            image = self.transform(image).float()
            gt = self.transform(np.expand_dims(gt, 2)).float()
            lr = self.transform(np.expand_dims(lr, 2)).float()

        sample = {'guidance': image, 'lr': lr, 'gt': gt, 'max': max_out, 'min': min_out, 'name': name}

        return sample