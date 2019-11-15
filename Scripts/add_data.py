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
#
# def get_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x
#
# file_path = '../Data/visdial_params.json'
#
# f = json.load(open(file_path, 'r'))
# itow = f['itow']
# img_info = f['img_train']
#
# img_list = []
# for i in img_info:
#     img_list.append(i['path'])
#
# updated_img_list = img_list[82000:]
# images = np.zeros((783,224,224,3))
# idx = 0
# for i in updated_img_list:
#     path = '../Data/'
#     path+=i
#     x = get_image(path)
#     images[idx:idx+1] = x
#
#     idx += 1
#
# img_height = 224
# img_width = 224
# base_model = VGG16(weights= 'imagenet', include_top=False, input_shape= (img_height,img_width,3))
#

# data = base_model.predict(images)
data = np.zeros((783,7,7,512))
if data.shape == (783,7,7,512):
    f = h5py.File('vgg16_data.hdf5', 'a')
    f['images_train'].resize(f['images_train'].shape[0] + data.shape[0], axis=0)
    f['images_train'][-data.shape[0]:] = data

    f.close()
