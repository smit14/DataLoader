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

file_path = '../Data/visdial_params.json'

f = json.load(open(file_path, 'r'))
itow = f['itow']
img_info = f['img_val']

img_list = []
for i in img_info:
    s = i['path']
    path = s[:13] +'train'+s[16:]
    img_list.append(path)

nums = 10
data = np.zeros((nums,7,7,512))
images = np.zeros((nums,224,224,3))
f = h5py.File('vgg16_val_data.hdf5','w')
f.create_dataset('images_val', data=data[0:1,:,:,:] , chunks=True, maxshape=(None,7,7,512))

img_height = 224
img_width = 224
base_model = VGG16(weights= 'imagenet', include_top=False, input_shape = (img_height,img_width,3))

t = time.time()

idx = 0
for i in img_list:
    path = '../Data/'
    path+=i
    x = get_image(path)
    images[idx:idx+1] = x

    idx += 1
    if idx%nums == 0:
        data = base_model.predict(images)
        f = h5py.File('vgg16_data.hdf5', 'a')
        f['images_val'].resize(f['images_val'].shape[0] + data.shape[0], axis=0)
        f['images_val'][-data.shape[0]:] = data
        idx = 0

        t = time.time() - t
        print(t / nums)
        t = time.time()

data = base_model(images[:idx,:,:,:])
f = h5py.File('vgg16_data.hdf5', 'a')
f['images_val'].resize(f['images_val'].shape[0] + data.shape[0], axis=0)
f['images_val'][-data.shape[0]:] = data
