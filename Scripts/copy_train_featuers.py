import h5py

new_path = '/home/smit/data_features.h5'
old_path = '/home/smit/val_features/val_features/vdl_img_vgg.h5'

newTemp = h5py.File(new_path, 'r')
oldTemp = h5py.File(old_path, 'r+')

newFile = newTemp['images_train']
oldFile = oldTemp['images_train']

total = 82000

i = 0
while(i<=total):
    oldFile[i,:,:,:] = newFile[i,:,:,:]
    i+=1
    if i%400==0:
        print(i)

newTemp.close()
oldTemp.close()
