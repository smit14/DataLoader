import numpy as np
import os
from urllib.request import urlopen,urlretrieve

from keras.models import load_model
from keras.utils import np_utils
from glob import glob
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
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
img_info = f['img_train']

img_list = []
for i in img_info:
    img_list.append(i['path'])

data = np.zeros((1000,7,7,512))
f = h5py.File('vgg16_data.hdf5','w')
f.create_dataset('images_train', data=data[0:1,:,:,:] , chunks=True, maxshape=(None,7,7,512))

img_height = 224
img_width = 224
base_model = VGG16(weights= 'imagenet', include_top=False, input_shape= (img_height,img_width,3))

t = time.time()

idx = 0
for i in img_list:
    path = '../Data/'
    path+=i
    x = get_image(path)
    data[idx,:,:,:] = base_model.predict(x)[0,:,:,:]
    idx += 1
    if idx%1000 == 0:
        f = h5py.File('vgg16_data.hdf5', 'a')
        f['images_train'].resize(f['images_train'].shape[0] + data.shape[0], axis=0)
        f['images_train'][-data.shape[0]:] = data
        idx = 0

        t = time.time() - t
        print(t / 1000)
