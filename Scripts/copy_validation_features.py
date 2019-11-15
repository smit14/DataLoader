import h5py

new_path = '/home/smit/val_features/vgg_val.hdf5'
old_path = '/home/smit/val_features/vdl_img_vgg.h5'

newTemp = h5py.File(new_path, 'r')
oldTemp = h5py.File(old_path, 'r+')

newFile = newTemp['images_val']
oldFile = oldTemp['images_val']

total = 40500

i = 1
while(i<=total):
    oldFile[i-1,:,:,:] = newFile[i,:,:,:]
    i+=1
    if i%200:
        print(i)

newTemp.close()
oldTemp.close()
