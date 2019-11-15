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

img_height = 224
img_width = 224
base_model = VGG16(weights= 'imagenet', include_top=False, input_shape= (img_height,img_width,3))

path = './290482.jpg'
x = get_image(path)
y = base_model.predict(x)
print(np.mean(y))

