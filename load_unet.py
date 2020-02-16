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


def show_img(img1, img2, img3, img4, tb=False, tb_dir=None):
    figure = plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(img1, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('origin')
    plt.subplot(1, 4, 2)
    plt.imshow(img2, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('detect')
    plt.subplot(1, 4, 3)
    plt.imshow(img3, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('denoise')
    plt.subplot(1, 4, 4)
    plt.imshow(img4, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('dilate_n')
    plt.show()
    if tb is True:
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_figure('images', figure, 0)
        writer.close()


def crop_roi(img1, mask):
    """
    Args:
        img1(torch.tensor[1, 1, 1024,1024]): input a img
        mask(torch.tensor[1, 1, 1024, 1024]): input a mask
    Returns:
        result_img(ndarray[1024,1024]): return a ROI
    """
    img1 = img1.squeeze().numpy()
    mask = mask.squeeze().int().numpy().astype(np.uint8)
    output = cv2.bitwise_and(img1, img1, mask=mask)
    return output


def get_roi(img):
    """
    Args:
        img(torch.tensor[1,1024,1024]): input a img
    Returns:
        result_img(ndarray[1024,1024]): return a ROI and mask
    """
    img = img.unsqueeze(0)  # tensor [1, 1, 1024, 1024]
    out = F.interpolate(img, size=[128, 128]).squeeze()  # tensor [128, 128]
    out2 = out.unsqueeze(0).unsqueeze(3).numpy()  # ndarray [1, 128, 128, 1]
    model_name = 'unet/trained/model_11.hdf5'
    model = load_model(model_name, custom_objects={'dice': train.dice, 'iou': train.iou})
    mask = model.predict(out2)[0][:,:,0] > 0.5  # ndarray [128, 128]
    mask2 = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0) # tensor [1, 1, 128, 128]
    large_mask = F.interpolate(mask2.float(), size=[1024, 1024])  # tensor [1, 1, 1024, 1024]
    result_img = crop_roi(img, large_mask)
    return result_img, large_mask.numpy()


def denoising(img):
    """
    https://blog.csdn.net/hjxu2016/article/details/77837765
    Args:
        img(ndarray[1024,1024]): input a img
    Returns:
        result_img(ndarray[1024,1024]): return a denoise img
    """
    kernel = np.ones((70, 70), np.uint8)
    kernel2 = np.ones((70, 70), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    dialation = cv2.dilate(result, kernel2, iterations=1)
    return result, dialation


roi, _ = get_roi(img)
de_roi, dia_roi = denoising(roi)
de_roi = torch.from_numpy(de_roi)
dia_roi = torch.from_numpy(dia_roi)
de_roi[de_roi < 0.05] = 0
de_roi[de_roi > 0] = 1
dia_roi[dia_roi < 0.05] = 0
dia_roi[dia_roi > 0] = 1

result = crop_roi(img, dia_roi)
# show_img(img.squeeze(), roi, de_roi, result, tb=False, tb_dir=logdir)


def findcontour(img):
    """
    https://blog.csdn.net/hjxu2016/article/details/77833336
    Args:
        img(ndarray[1024,1024]): gray img
    Returns:
        countours
    """
    img2 = np.array(img * 255, dtype=np.uint8)
    ret, binary = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


countours, hierarchy = findcontour(dia_roi)


def crop_by_countours(origin_img, countours):
    """
    https://blog.csdn.net/hjxu2016/article/details/77833336
    Args:
        origin_img(torch.tensor[1~c, 1024~h,1024~w]): img
        countours: 2d nested list, represent left and right lungs
    Returns:
        img1 : torch.tensor[h, w]
        img2 : torch.tensor[h, w]
    """
    c1 = countours[0].reshape(-1, 2)
    c2 = countours[1].reshape(-1, 2)
    # result_c = result.copy()
    # for point in c1:
    #     cv2.circle(result_c, tuple(point), 5, (255, 0, 0), -1)
    #
    # for point in c2:
    #     cv2.circle(result_c, tuple(point), 5, (255, 0, 0), -1)
    rect1 = cv2.boundingRect(c1)
    # print(rect1)
    x1, y1, w1, h1 = rect1
    # cv2.rectangle(result_c, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)

    rect2 = cv2.boundingRect(c2)
    # print(rect2)
    x2, y2, w2, h2 = rect2
    # cv2.rectangle(result_c, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)

    origin_img = origin_img.squeeze()
    img1 = origin_img[y1:y1+h1, x1:x1+w1]
    img2 = origin_img[y2:y2+h2, x2:x2+w2]
    print(type(img1))
    return img1, img2


img1, img2 = crop_by_countours(img, countours)