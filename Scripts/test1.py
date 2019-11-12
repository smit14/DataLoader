import time
import json
import h5py

idx = 4

file_path = 'vgg16_data.hdf5'
f = h5py.File(file_path, 'r')
imgs = f['images_train']
img = imgs[idx]
print(img[:,:,0])
f.close()
