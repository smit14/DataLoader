import h5py
import numpy as np
idx = 2
copied_path = './vdl_img_vgg.h5'
fd = h5py.File(copied_path, 'r')

imgs = fd['images_train']

img = imgs[idx,:,:,:]

print(img[:,:,0])
print(np.mean(img))

fd.close()
