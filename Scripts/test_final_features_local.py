import numpy as np
import json
import h5py


path = '/home/smit/val_features/val_features/vdl_img_vgg.h5'

f = h5py.File(path, 'r')
train_feats = f['images_train']
val_feats = f['images_val']

idx_list = [0,5,50,4335,2134]

print('training features...')
for i in idx_list:
    print(np.mean(train_feats[i,:,:,:]))

print('validation features...')
for i in idx_list:
    print(np.mean(val_feats[i,:,:,:]))
