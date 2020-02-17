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
from PIL import Image
from PIL import ImageEnhance


root = "/data15/boxi/anomaly detection/data/"


class AddContrast(object):
    def __call__(self, sample):
        enh_con = ImageEnhance.Contrast(sample)
        contrast = 3.0
        img_contrasted = enh_con.enhance(contrast)
        return img_contrasted


transform = transforms.Compose(
    [
        AddContrast(),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ]
)
dataset = PneumoniaDataset(root=root, train=False, transform=transform)
img, landmark = dataset[5] # PIL.Image.Image mode=RGB size=1024*1024


# 查看图片
logdir = "/data15/boxi/anomaly detection/runs/resizeImage/" + datetime.now().strftime("%Y%m%d-%H%M%S")



def show_img(img1, img2, img3, img4, img5, img6, tb=False, tb_dir=None):
    figure = plt.figure()
    plt.subplot(1, 6, 1)
    plt.imshow(img1, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('origin')
    plt.subplot(1, 6, 2)
    plt.imshow(img2, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('roi')
    plt.subplot(1, 6, 3)
    plt.imshow(img3, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('de_b_noise')
    plt.subplot(1, 6, 4)
    plt.imshow(img4, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('erosion')
    plt.subplot(1, 6, 5)
    plt.imshow(img5, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('de_w_noise')
    plt.subplot(1, 6, 6)
    plt.imshow(img6, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('dialation')
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    de_b_noise = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 去前景黑点
    er = cv2.erode(de_b_noise, kernel, iterations=1) # 缩小边缘
    de_w_noise = cv2.morphologyEx(er, cv2.MORPH_OPEN, kernel) # 去前景白点
    dialation = cv2.dilate(de_w_noise, kernel, iterations=2) # 扩大

    return de_b_noise, er, de_w_noise, dialation


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
    areaArray = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    # [(area, array), (area, array), ....]
    a = sorteddata[0:2]
    b = [i[1] for i in a]

    return b, hierarchy


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

    # ori_c = origin_img.squeeze().numpy().copy()

    rect1 = cv2.boundingRect(c1)
    # print(rect1)
    # x1, y1, w1, h1 = rect1
    # cv2.rectangle(ori_c, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

    rect2 = cv2.boundingRect(c2)
    # print(rect2)
    # x2, y2, w2, h2 = rect2
    # cv2.rectangle(ori_c, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

    # origin_img = origin_img.squeeze()
    # img1 = origin_img[y1:y1+h1, x1:x1+w1]
    # img2 = origin_img[y2:y2+h2, x2:x2+w2]
    #
    # show_img(origin_img.squeeze(), ori_c, img1, img2)
    return rect1, rect2


def pack_up_all(img):
    roi, _ = get_roi(img)
    # 转为0,1 mask
    # roi[roi < 0.01] = 0
    # roi[roi > 0] = 1
    
    de_b_noise, er, de_w_noise, dialation = denoising(roi)

    de_b_noise = torch.from_numpy(de_b_noise)
    er = torch.from_numpy(er)
    de_w_noise = torch.from_numpy(de_w_noise)
    dialation = torch.from_numpy(dialation)

    show_img(img.squeeze(), roi, de_b_noise, er, de_w_noise, dialation)
    contours, hierarchy = findcontour(dialation)
    rect1, rect2 = crop_by_countours(img, contours)
    return rect1, rect2


rect1, rect2 = pack_up_all(img)
print(rect1)
print(rect2)