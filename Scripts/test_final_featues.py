import numpy as np
import os
from urllib.request import urlopen,urlretrieve

from keras.models import load_model
from keras.utils import np_utils
from glob import glob
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import time
import json
import h5py

def get_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

train_path = [{'path': 'train2014/COCO_train2014_000000378466.jpg', 'imgId': 378466},
 {'path': 'train2014/COCO_train2014_000000012597.jpg', 'imgId': 12597},
 {'path': 'train2014/COCO_train2014_000000082990.jpg', 'imgId': 82990},
 {'path': 'train2014/COCO_train2014_000000473797.jpg', 'imgId': 473797},
 {'path': 'train2014/COCO_train2014_000000318132.jpg', 'imgId': 318132}]

val_path = [{'path': 'val2014/COCO_val2014_000000378467.jpg', 'imgId': 378467},
 {'path': 'val2014/COCO_val2014_000000256903.jpg', 'imgId': 256903},
 {'path': 'val2014/COCO_val2014_000000119861.jpg', 'imgId': 119861},
 {'path': 'val2014/COCO_val2014_000000514089.jpg', 'imgId': 514089},
 {'path': 'val2014/COCO_val2014_000000209274.jpg', 'imgId': 209274}]

train_list = []
val_list = []

for i in range(5):
    tr = train_path[i]['path']
    vl = val_path[i]['path']
    train_list.append(tr)
    val_list.append(vl)

img_height = 224
img_width = 224
base_model = VGG16(weights= 'imagenet', include_top=False, input_shape = (img_height,img_width,3))

print('training features....')
for i in train_list:
    path = '../Data/'
    path+=i
    x = get_image(path)
    y = base_model.predict(x)
    print(np.mean(y))

print('validation features...')
for i in val_list:
    path = '../Data/'
    path+=i
    x = get_image(path)
    y = base_model.predict(x)
    print(np.mean(y))
