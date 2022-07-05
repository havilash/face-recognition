import sys
sys.path.append('')

from constants import *

import shutil
import os
import time
import uuid
import cv2

import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import numpy as np

from PIL import Image
import albumentations as alb

# Avoid OOM errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)

# tf.config.list_physical_devices('GPU')


def create_images(n):
    number_images = n

    cap = cv2.VideoCapture(0)
    for imgnum in range(number_images):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(DATA_PATH, 'images', f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # create labels with labelme

def create_augmented_data(n):
    augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                            alb.HorizontalFlip(p=0.5), 
                            alb.RandomBrightnessContrast(p=0.2),
                            alb.RandomGamma(p=0.2), 
                            alb.RGBShift(p=0.2), 
                            alb.VerticalFlip(p=0.5)], 
                            bbox_params=alb.BboxParams(
                                format='albumentations', 
                                label_fields=['class_labels']))

    for image in os.listdir(os.path.join(DATA_PATH, 'images')):
        img = cv2.imread(os.path.join(DATA_PATH, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join(DATA_PATH, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]

            img_shape = img.shape
            coords = list(np.divide(coords, [img_shape[1], img_shape[0], img_shape[1], img_shape[0]]))

        try: 
            for x in range(n):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join(AUG_DATA_PATH, 'images', f'{image.split(".")[0]}.{x}.jpg'), cv2.resize(augmented['image'], IMAGE_SIZE))

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0 


                with open(os.path.join(AUG_DATA_PATH, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)


def remove_data(path):
    for f in ['images', 'labels']:
        p = os.path.join(path, f)
        for file in os.listdir(p):
            os.remove(os.path.join(p, file))
            print(f"removing {os.path.join(p, file)} ...")
    print("removing completed")

def create_train_test_data():
    images = list(map(lambda x: os.path.join(AUG_DATA_PATH, 'images', x), os.listdir(os.path.join(AUG_DATA_PATH, 'images'))))
    labels = list(map(lambda x: os.path.join(AUG_DATA_PATH, 'labels', x), os.listdir(os.path.join(AUG_DATA_PATH, 'labels'))))

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.7, shuffle=False)
    test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels, test_size=0.5, shuffle=False)
    
    for f in train_images:
        shutil.copy(f, os.path.join(AUG_DATA_PATH, 'train', 'images'))
    for f in train_labels:
        shutil.copy(f, os.path.join(AUG_DATA_PATH, 'train', 'labels'))

    for f in test_images:
        shutil.copy(f, os.path.join(AUG_DATA_PATH, 'test', 'images'))
    for f in test_labels:
        shutil.copy(f, os.path.join(AUG_DATA_PATH, 'test', 'labels'))
    
    for f in val_images:
        shutil.copy(f, os.path.join(AUG_DATA_PATH, 'val', 'images'))
    for f in val_labels:
        shutil.copy(f, os.path.join(AUG_DATA_PATH, 'val', 'labels'))
        

def load_data():
    def load_image(x): 
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        return img

    train_images = tf.data.Dataset.list_files(os.path.join(AUG_DATA_PATH, 'train', 'images', '*.jpg'), shuffle=False)
    train_images = train_images.map(load_image)
    train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
    train_images = train_images.map(lambda x: x/255)

    test_images = tf.data.Dataset.list_files(os.path.join(AUG_DATA_PATH, 'test', 'images', '*.jpg'), shuffle=False)
    test_images = test_images.map(load_image)
    test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
    test_images = test_images.map(lambda x: x/255)

    val_images = tf.data.Dataset.list_files(os.path.join(AUG_DATA_PATH, 'val', 'images', '*.jpg'), shuffle=False)
    val_images = val_images.map(load_image)
    val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
    val_images = val_images.map(lambda x: x/255)


    def load_labels(label_path):
        with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
            label = json.load(f)
            
        return [label['class']], label['bbox']

    
    train_labels = tf.data.Dataset.list_files(os.path.join(AUG_DATA_PATH, 'train', 'labels', '*.json'), shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    test_labels = tf.data.Dataset.list_files(os.path.join(AUG_DATA_PATH, 'test', 'labels', '*.json'), shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    val_labels = tf.data.Dataset.list_files(os.path.join(AUG_DATA_PATH, 'val', 'labels', '*.json'), shuffle=False)
    val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    train_labels.as_numpy_iterator().next()

    return (train_images, train_labels, test_images, test_labels, val_images, val_labels)

if __name__ == "__main__":
    # create_images(30)
    # create_augmented_data(60)
    # remove_data(AUG_DATA_PATH)

    # create_train_test_data()
    pass
