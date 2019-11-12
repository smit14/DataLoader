import h5py
import numpy as np
import time

f2 = h5py.File('vgg16_data.hdf5','r')
images = f2['train_images']

temp_data = images[1:2,:,:,:]
f = h5py.File('vgg16_real_data.hdf5','w')
f.create_dataset('images_train',data = temp_data, chunks=True, maxshape=(None,7,7,512))

f2 = h5py.File('vgg16_data.hdf5','r')
images = f2['images_train']
t = time.time()
img = np.zeros(1000,7,7,512)

for i in range(images.shape[0]-2)/1000:
    s = i*1000+2
    e = min((i+1)*1000+2,images.shape[0])
    data = images[s:e,:,:,:]
    f = h5py.File('vgg16_real_data.hdf5', 'a')
    f['images_train'].resize(f['images_train'].shape[0] + data.shape[0], axis=0)
    f['images_train'][-data.shape[0]:] = data
    print(i)
    t = time.time() - t
    print(t / 1000)
    t = time.time()



