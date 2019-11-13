import h5py
import time
new_path = '../vgg16_data.hdf5'
old_path = '/vdl_img_vgg.h5'

newTemp = h5py.File(new_path, 'r')
oldTemp = h5py.File(old_path, 'r+')


newFile = newTemp['images_train']
oldFile = oldTemp['images_train']

total = 82000

i = 1
t = time.time()
while(i<=total):
    oldFile[i-1,:,:,:] = newFile[i,:,:,:]
    print(i)
    i+=1

newTemp.close()
oldTemp.close()