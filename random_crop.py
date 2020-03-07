"""
croped dataset:
crop BATCH_SIZE small images from 1000*1000 origin images

trainset: 7080 origin
validationset: 771 origin
testnormal: 1000 origin
testanomly:

"""
import pandas as pd
from PIL import Image
from dataset import dcm_loader
import matplotlib.pyplot as plt
import random
import os

DATA_DIR = "/data15/boxi/anomaly detection/data/"
BATCH_SIZE = 64
CROPED_SIZE = 64

def get_paths_from_csv(path):
    """
    retrive paths list from csv file
    :return: list
    """
    df = pd.read_csv(str(path))
    if df is not None:
        return df.path


def crop(path, name):
    """
    :param path: path of an image
    :param name: origin image path
    :return: none
    """
    img = dcm_loader(path)
    left, right, top, bottom = 120, 900, 200, 800
    img1 = img.crop((left, top, right, bottom))
    print(img1.size)  # (780, 600)
    name = name.split(".")[0]

    try:
        for i in range(BATCH_SIZE):
            left = random.randint(0, 780-CROPED_SIZE)
            top = random.randint(0, 600-CROPED_SIZE)
            img_crop = img.crop((left, top, left + CROPED_SIZE, top + CROPED_SIZE))
            img_crop.save("%safter-crop/trainset/%s-%d.png" % (DATA_DIR, name, i))
    except Exception as e:
        print("Exception occur at crop %s" % name)
        print(e)






if __name__ == '__main__':
    train_data_paths = get_paths_from_csv(DATA_DIR + 'train-normal-data.csv')
    print(len(train_data_paths))  # 7080
    test_data_paths = get_paths_from_csv(DATA_DIR + 'test-normal-data.csv')
    print(len(test_data_paths))  # 1771
    crop(DATA_DIR + 'pneumonia_data/stage_2_train_images/' + train_data_paths[0], train_data_paths[0])
    for i in range(len(train_data_paths)):
        crop(DATA_DIR + 'pneumonia_data/stage_2_train_images/' + train_data_paths[i], train_data_paths[i])

