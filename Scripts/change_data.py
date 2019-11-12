import h5py
import numpy as np
import time

f = h5py.File('vgg16_real_data.hdf5','w')
f.create_dataset('images_train',chunks=True, maxshape=(None,7,7,512))

f2 = h5py.File('vgg16_data.hdf5','r')
images = f2['train_images']
t = time.time()
img = np.zeros(1000,7,7,512)

for i in range(images.shape[0])/1000:
    s = i*1000+1
    e = min((i+1)*1000+1,images.shape[0])
    data = images[s:e,:,:,:]
    f = h5py.File('vgg16_real_data.hdf5', 'a')
    f['images_train'].resize(f['images_train'].shape[0] + data.shape[0], axis=0)
    f['images_train'][-data.shape[0]:] = data
    print(i)
    t = time.time() - t
    print(t / 1000)
    t = time.time()



