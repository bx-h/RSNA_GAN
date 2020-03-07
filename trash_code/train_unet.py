import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa

import skimage.io
import skimage.measure
from tqdm import tqdm
from PIL import Image

import requests
import shutil
import zipfile

import mdai
mdai.__version__

mdai_client = mdai.Client(domain='public.md.ai', access_token="3de55d83f2705ec4159cc62926382533")
p = mdai_client.project('aGq4k6NW', path='../lesson2-data')

# download MD.ai's dilated unet implementation
UNET_URL = 'https://s3.amazonaws.com/md.ai-ml-lessons/unet.zip'
UNET_ZIPPED = 'unet.zip'

if not os.path.exists(UNET_ZIPPED):
    r = requests.get(UNET_URL, stream=True)
    if r.status_code == requests.codes.ok:
        with open(UNET_ZIPPED, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    else:
        r.raise_for_status()

    with zipfile.ZipFile(UNET_ZIPPED) as zf:
        zf.extractall()

p.show_label_groups()

# this maps label ids to class ids as a dict obj
labels_dict = {'L_A8Jm3d':1 # Lung
              }

print(labels_dict)
p.set_labels_dict(labels_dict)

p.show_datasets()

dataset = p.get_dataset_by_id('D_rQLwzo')
dataset.prepare()

image_ids = dataset.get_image_ids()
len(image_ids)

imgs_anns_dict = dataset.imgs_anns_dict

from unet import dataset
from unet import dilated_unet
from unet import train

from datetime import datetime
import tensorflow as tf
import io
from torch.utils.tensorboard import SummaryWriter


images, masks = dataset.load_images(imgs_anns_dict)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

CONFIG_FP = 'unet/configs/11.json'
name = os.path.basename(CONFIG_FP).split('.')[0]
print(name)

with open(CONFIG_FP, 'r') as f:
    config = json.load(f)

# increase the number of epochs for better prediction
history = train.train(config, name, images,masks, num_epochs=40)

# -------------------------------------- see the train process-----------------------
def image_grid():
    figure = plt.figure()
    plt.plot(history.history['accuracy'], 'orange', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'blue', label='Validation accuracy')
    plt.plot(history.history['loss'], 'red', label='Training loss')
    plt.plot(history.history['val_loss'], 'green', label='Validation loss')
    plt.legend()
    plt.show()
    return figure


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


logdir = "/data15/boxi/anomaly detection/runs/unetplots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=logdir)
grid = image_grid()
writer.add_figure('images', grid, 0)
writer.close()
