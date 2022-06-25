from constants import *
import data

import tensorflow as tf
from sklearn.model_selection import train_test_split

import json
import numpy as np
from matplotlib import pyplot as plt

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')


def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = tf.data.Dataset.list_files(os.path.join(AUG_DATA_PATH, 'images', '*.jpg'), shuffle=False)
images = images.map(load_image)
images = images.map(lambda x: x/255)

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']

labels = tf.data.Dataset.list_files(os.path.join(AUG_DATA_PATH, 'labels', '*.json'), shuffle=False)
labels = labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.7, shuffle=False)
test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels, test_size=0.15, shuffle=False)