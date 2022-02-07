
import os
import glob
import random
import numpy as np
from torch.utils import data
from torchvision import transforms as trans
from PIL import Image

class ImageDataset(data.Dataset):

    def __init__(self, dataset_args,train=True,paired=True):
        # affine
        # randomcropsize
        # mae mask
        # D struct  pretrain resnet
        # task loss

        self.root=dataset_args.data_root
        self.train=train   # train or val
        self.split=dataset_args.split # the number of training images e.g. 65000 training imgs for ffhq
        #self.unaligned =False  # paired(False) or unpaired (True)
        self.paired=paired
        self.transform = trans.ToTensor()
        self.transform_t=trans.Compose([
            trans.ToTensor(),
            trans.RandomApply([trans.RandomAffine(degrees=(-30, 30))], p=0.5),
            # trans.RandomApply([trans.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0))], p=0.5),
            # trans.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))
        ])
        # self.files_s = sorted(glob.glob(os.path.join(self.root, '/Source') + '/*.*'))   # get the whole files in directory
        # self.files_t = sorted(glob.glob(os.path.join(self.root, '/Target') + '/*.*'))
        self.files_s=sorted(glob.glob(self.root+'/*.*'))
        self.files_t=sorted(glob.glob(self.root+'/*.*'))
        self.source_list = self.collect_image(source=True)
        self.target_list = self.collect_image(source=False)
        self.max_val = dataset_args.max_val
        self.min_val = dataset_args.min_val
        self.size = dataset_args.size

    def collect_image(self,source=True):
        image_path_list = []
        if source:
            if self.train:
                image_path_list=self.files_s[:int(self.split)]
            else:
                image_path_list = self.files_s[int(self.split):]
        else:
            if self.train:
                image_path_list = self.files_t[:int(self.split)]
            else:
                image_path_list=self.files_t[int(self.split):]
        return image_path_list

    def __getitem__(self, index):
        item_s = self.transform(Image.open(self.source_list[index % len(self.files_s)]))
        if self.paired:
            item_t = self.transform_t(Image.open(self.target_list[index % len(self.files_t)]))
        else:
            item_t = self.transform_t(Image.open(self.target_list[random.randint(0, len(self.files_t) - 1)]))

        item_s=item_s*(self.max_val - self.min_val) + self.min_val
        item_t=item_t*(self.max_val - self.min_val) + self.min_val
        return {'x_s': item_s, 'x_t': item_t}

    def __len__(self):
        return max(len(self.source_list), len(self.target_list))

class CelebaHQ(data.Dataset):

    def __init__(self, dataset_args, train=True):
        self.name = 'CelebaHQ'
        self.data_root = dataset_args.data_root
        self.train_root = os.path.join(self.data_root, 'train')
        self.val_root = os.path.join(self.data_root, 'val')
        self.train = train
        self.train_list = self.collect_image(self.data_root)
        self.val_list = self.collect_image(self.data_root)
        # self.train_list=self.collect_image(self.data_root)
        # self.train_list=self.collect_image(self.data_root)
        self.transform = trans.ToTensor()
        self.max_val = dataset_args.max_val
        self.min_val = dataset_args.min_val
        self.size = dataset_args.size

    def collect_image(self, root):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            for file_name in sorted(os.listdir(split_root)):
                file_path = os.path.join(split_root, file_name)
                image_path_list.append(file_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size=256):
        # the imagesize of ffhq dataset is 256
        # img=image
        img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        if self.train:
            image_path = self.train_list[index]
        else:
            image_path = self.val_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img, size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.val_list)
class Lsun_tower(data.Dataset):
    def __init__(self, dataset_args, train=True):
        self.name = 'tower'
        self.data_root = dataset_args.data_root
        self.train = train
        self.split=dataset_args.split
        self.train_list = self.collect_image(self.data_root)
        self.val_list = self.collect_image(self.data_root)
        # self.transform = trans.Compose([trans.CenterCrop(256),trans.ToTensor()])
        self.transform = trans.Compose([trans.Resize([256,256]), trans.ToTensor()])
        self.max_val = dataset_args.max_val
        self.min_val = dataset_args.min_val
        self.size = dataset_args.size
    def collect_image(self, root):
        image_path_list = []
        img_list = os.listdir(root)
        if self.train:
            train_list = sorted(img_list)[:self.split]
            for file_name in ((train_list)):
                file_path = os.path.join(root, file_name)
                image_path_list.append(file_path)
        else:
            val_list = sorted(img_list)[:self.split]
            for file_name in ((val_list)):
                file_path = os.path.join(root, file_name)
                image_path_list.append(file_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size=256):
        # the imagesize of ffhq dataset is 256
        # img = image
        img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        if self.train:
            image_path = self.train_list[index]
        else:
            image_path = self.val_list[index]
        img = self.read_image(image_path)
        #img = self.resize_image(img, size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.val_list)

class FFHQDataset(data.Dataset):

    def __init__(self, dataset_args, train=True):
        self.name = 'FFHQ'
        self.data_root = dataset_args.data_root
        self.train = train
        self.split=dataset_args.split
        self.train_list = self.collect_image(self.data_root)
        self.val_list = self.collect_image(self.data_root)
        self.transform = trans.ToTensor()
        self.max_val = dataset_args.max_val
        self.min_val = dataset_args.min_val
        self.size = dataset_args.size
    def collect_image(self, root):
        image_path_list = []
        img_list = os.listdir(root)
        if self.train:
            train_list = sorted(img_list)[:self.split]
            for file_name in ((train_list)):
                file_path = os.path.join(root, file_name)
                image_path_list.append(file_path)
        else:
            val_list = sorted(img_list)[self.split:]
            for file_name in ((val_list)):
                file_path = os.path.join(root, file_name)
                image_path_list.append(file_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size=256):
        # the imagesize of ffhq dataset is 256
        img = image
        # img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        if self.train:
            image_path = self.train_list[index]
        else:
            image_path = self.val_list[index]
        img = self.read_image(image_path)
        #img = self.resize_image(img, size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.val_list)


if __name__ == '__main__':
    class Config:
        data_root = '/mnt/ssd2/xintian/datasets/celeba_hq/'
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = 256


    config = Config()
    dataset = CelebaHQ(config)
    for i, data in enumerate(dataset):
        print(data.shape, data.max(), data.min())
        break
    print(dataset.__len__())
