
from __future__ import print_function, division
import numpy as np
from os import listdir
from random import shuffle
import pydicom as dicom
from skimage.transform import resize

# find unique data regardless of the file prefix
def file_list_info(data_path, batch_size=1):
    files = sorted([f for f in listdir(data_path) if f.endswith('_GT.IMA')])
    nfiles = len(files)
    batches_per_epoch = nfiles // batch_size
    gt_path = '%s/%s'%(data_path,files[0])  #Read the first GT image and return its size
    size = np.shape(dicom.dcmread(gt_path).pixel_array)
    img_size = (size[0], size[1], 3)
    return (files, batches_per_epoch, img_size)

def generator_3dCoil(data_path, img_size, testing = False, shuffle_epoch=True):
    files, batches_per_epoch, a = file_list_info(data_path)
    x = np.zeros((1,)+(img_size[0], img_size[1], 3))
    y = np.zeros((1,)+(img_size[0], img_size[1], 1))

    while True:       
        if shuffle_epoch:
            shuffle(list(files))
        else:
            files = sorted(files)
        
        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(1):

                file_ind = batch_cnt*1+file_cnt
                files=list(files)
                gt_path = '%s/%s'%(data_path,files[file_ind])
                im1_path = gt_path[:-6] + 'CH1.IMA'
                im2_path = gt_path[:-6] + 'CH2.IMA'
                im3_path = gt_path[:-6] + 'CH3.IMA' 

                ch1 = 2**16 * resize(dicom.dcmread(im1_path).pixel_array, (img_size[0], img_size[1]), anti_aliasing=True)
                ch2 = 2**16 * resize(dicom.dcmread(im2_path).pixel_array, (img_size[0], img_size[1]), anti_aliasing=True)
                ch3 = 2**16 * resize(dicom.dcmread(im3_path).pixel_array, (img_size[0], img_size[1]), anti_aliasing=True)
                gt = 2**16 * resize(dicom.dcmread(gt_path).pixel_array, (img_size[0], img_size[1]), anti_aliasing=True)

                x[file_cnt,...,0] = ch1
                x[file_cnt,...,1] = ch2
                x[file_cnt,...,2] = ch3
                
                # print('\n', np.nanmax(ch1), '\n',np.nanmax(ch2), '\n',np.nanmax(ch3), '\n',np.nanmax(gt))

                y[file_cnt,...,0] = gt
                
                fname = files[file_ind]
                
#            print(im1_path, '\n', im2_path, '\n', im3_path, '\n', gt_path) 
#                print(ch1.shape)                
#                print(y.shape)

            if testing is False:
                yield (x, y)
            else:
                fname = files[file_ind]
                yield (x, y, fname)
