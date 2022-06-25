import os
import time
import uuid
import cv2

# import tensorflow as tf
import json
import numpy as np
# from matplotlib import pyplot as plt

from PIL import Image
import albumentations as alb

# Avoid OOM errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)

# tf.config.list_physical_devices('GPU')


IMAGES_PATH = os.path.join('data','images')
IMAGE_SIZE = (120, 120)


def create_images(n):
    number_images = n

    cap = cv2.VideoCapture(0)
    for imgnum in range(number_images):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        frame = cv2.resize(frame, (120, 120))
        imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # labelme

def create_augmented_images():
    augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                            alb.HorizontalFlip(p=0.5), 
                            alb.RandomBrightnessContrast(p=0.2),
                            alb.RandomGamma(p=0.2), 
                            alb.RGBShift(p=0.2), 
                            alb.VerticalFlip(p=0.5)], 
                            bbox_params=alb.BboxParams(
                                format='albumentations', 
                                label_fields=['class_labels']))

    for image in os.listdir(os.path.join('data', 'images')):
        img = cv2.imread(os.path.join('data', 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            img_shape = img.shape
            coords = list(np.divide(coords, [*img_shape, *img_shape]))

        try: 
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

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


                with open(os.path.join('aug_data', 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    create_images(1)
