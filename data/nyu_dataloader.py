import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from data.common import get_patch,arugment

class NYU_v2_dataset(Dataset):

    def __init__(self,root_dir,scale=4,train=True,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train

        if train:
            self.depths = np.load('%s/train_depth_split.npy' % root_dir)
            self.images = np.load('%s/train_images_split.npy' % root_dir)

        else:
            self.depths = np.load('%s/test_depth.npy' % root_dir)
            self.images = np.load('%s/test_images_v2.npy' % root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        if self.train:
            image, depth = get_patch(img=image,gt=np.expand_dims(depth,2),patch_size=256)
            image, depth = arugment(img=image,gt=depth)

        h,w = depth.shape[:2]
        s = self.scale
        lr = np.array(Image.fromarray(depth.squeeze()).resize((w // s, h // s), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            lr = self.transform(np.expand_dims(lr,2)).float()

        sample = {'guidance': image,  'lr': lr, 'gt': depth}

        return sample


