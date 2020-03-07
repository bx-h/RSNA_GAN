import os
import sys

import pandas as pd
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import SimpleITK as sitk

import pydicom
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter


class PneumoniaDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.join(root, "pneumonia_data")
        self.transform = transform
        self.train = train

        self.image_data_dir = os.path.join(self.root, 'stage_2_train_images')
        iter_fold = 1
        self.imgs_path, self.targets = self.get_data(iter_fold, os.path.join(self.root, 'split_data'))

        self.loader = dcm_loader
        classes_name = ['Normal', 'Lung Opacity', '‘No Lung Opacity/Not Normal']
        self.classes = list(range(len(classes_name)))
        self.target_img_dict = dict()
        targets = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(targets == target)[0]
            self.target_img_dict.update({target: indexes})

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.imgs_path[index]
        target = self.targets[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path

    def __len__(self):
        return len(self.targets)

    def get_data(self, iterNo, data_dir):

        if self.train:
            csv = 'pneumonia_split_{}_train.csv'.format(iterNo)
        else:
            csv = 'pneumonia_split_{}_test.csv'.format(iterNo)

        fn = os.path.join(data_dir, csv)
        csvfile = pd.read_csv(fn, index_col=0)
        raw_data = csvfile.values

        data = []
        targets = []
        for (path, label) in raw_data:
            data.append(os.path.join(self.image_data_dir, path))
            targets.append(label)

        return data, targets


def dcm_loader(path):
    ds = sitk.ReadImage(path)
    img_array = sitk.GetArrayFromImage(ds)
    img_bitmap = Image.fromarray(img_array[0]).convert('RGB')
    return img_bitmap


# def print_dataset(dataset, print_time):
#     print(len(dataset))             # train: 21345 test: 5339
#     from collections import Counter
#     counter = Counter()
#     labels = []
#     images = []
#     for index, (img, label) in enumerate(dataset):
#         if index % print_time == 0:
#             print(img.size(), label)  # torch.Size([3, 1024, 1024]) 2/1/0
#             images.append(img)        # img :torch.Tensor
#             print(type(img))
#
#         labels.append(label)          # labels: [0, 0, 1, 1, 0, 2, ...]
#     counter.update(labels)            # train: counters: {2: 9456, 0: 7080, 1: 4809}
#     print(counter)                    # test: counters: {2: 2365, 0: 1771, 1: 1203 }

    # 保存图片
    # p_image = images[0].permute(1, 2, 0)
    # print(p_image.size())
    # p_image = p_image.numpy()
    # outpath = "/data15/boxi/anomaly detection/code/RSNA_GAN/patient2.jpg"
    # plt.imsave(outpath, p_image)

    logdir = "/data15/boxi/anomaly detection/runs/"
    writer = SummaryWriter(log_dir=logdir)
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.close()


if __name__ == "__main__":
    root = "/data15/boxi/anomaly detection/data/"
#    dataset = PneumoniaDataset(root=root, train=True, transform=transforms.ToTensor())
#    print_dataset(dataset, print_time=10000)

    dataset = PneumoniaDataset(root=root, train=False, transform=transforms.ToTensor())
    # print_dataset(dataset, print_time=1000)
