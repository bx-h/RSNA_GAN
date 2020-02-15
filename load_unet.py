import os
import matplotlib.pyplot as plt
from keras.models import load_model
from unet import train
import keras.backend as K
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from datetime import datetime
import numpy as np
import cv2
import torch

from Pneumonia import PneumoniaDataset
from torchvision import transforms

root = "/data15/boxi/anomaly detection/data/"

transform = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.ToTensor()
    ]
)
dataset = PneumoniaDataset(root=root, train=True, transform=transform)
img, _ = dataset[0]

# 查看图片
logdir = "/data15/boxi/anomaly detection/runs/resizeImage/" + datetime.now().strftime("%Y%m%d-%H%M%S")


def show_img(img, tb=False, tb_dir=None):
    plt.imshow(img, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('finish')
    plt.show()
    if tb is True:
        writer = SummaryWriter(log_dir=tb_dir)
        grid = torchvision.utils.make_grid(img)
        writer.add_image('images', grid, 0)
        writer.close()


def get_roi(img):
    """
    Args:
        img(torch.tensor[1,1024,1024]): input a img
    Returns:
        result_img(ndarray[1024,1024]): return a mask from
    """
    img = img.unsqueeze(0)  # tensor [1, 1, 1024, 1024]
    out = F.interpolate(img, size=[128, 128]).squeeze()  # tensor [128, 128]
    out2 = out.unsqueeze(0).unsqueeze(3).numpy()  # ndarray [1, 128, 128, 1]
    model_name = 'unet/trained/model_11.hdf5'
    model = load_model(model_name, custom_objects={'dice': train.dice, 'iou': train.iou})
    mask = model.predict(out2)[0][:,:,0] > 0.5  # ndarray [128, 128]
    mask2 = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0) # tensor [1, 1, 128, 128]
    large_mask = F.interpolate(mask2.float(), size=[1024, 1024]).squeeze().int()  # tensor [1024, 1024]
    img = img.squeeze().numpy()
    result_img = cv2.bitwise_and(img, img, mask=large_mask.numpy().astype(np.uint8))
    return result_img


roi = get_roi(img)
roi = torch.from_numpy(roi)
show_img(roi, tb=True, tb_dir=logdir)

# mask:
# [[ True  True  True ... False False False]
#  [ True  True  True ... False False False]
#  [ True  True  True ... False False False]
#  ...
#  [ True  True  True ... False False False]
#  [ True  True  True ... False False False]
#  [ True  True  True ... False False False]]

