import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--idx', default=0)
opt = parser.parse_args()

idx = int(opt.idx)
copied_path = './vdl_img_vgg.h5'
fd = h5py.File(copied_path, 'r')

imgs = fd['images_train']

img = imgs[idx,:,:,:]

print(img[:,:,0])
print(np.mean(img))

fd.close()
