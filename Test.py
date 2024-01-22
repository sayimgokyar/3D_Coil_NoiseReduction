"""Prepared by sayim gokyar @January 21st for testing the denoising performance of 3D coil data
"""
from __future__ import print_function, division

import os
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras import backend as K
from utils.generator_3dCoil import file_list_info, generator_3dCoil
from utils.models import unet_2d

print(__doc__)
#model_weights = './weights/PhantomTrainedWeights.h5'
model_weights = './weights/Sub01TrainedWeights.h5'

lyrs=2 #number of Unet Layers
conv_filters= 96

# Specify directories
#test_path = './MRIs/Phantom02'
#test_path = './MRIs/Sub01_02_03_processed'

test_path = './MRIs/Sub02_03_processed'
test_result_path = './results/SubjectTrainedWeightPredictions'
save_file = True

ch1_peak = []
ch1_noise = []
ch1_SNRmax = []
ch1_SNRave = []

pred_peak = []
pred_noise = []
pred_SNRmax = []
pred_SNRave = []

ref_peak = []
ref_noise = []
ref_SNRmax = []
ref_SNRave = []

FNAME = []
def test_denoising(test_result_path, test_path, save_file, model_weights):
    img_cnt = 0
    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')

    # Read the files that will be tested by trained W 
    test_files, ntest, img_size = file_list_info(test_path)
    model = unet_2d(img_size, lyrs, conv_filters)
    model.load_weights(model_weights);
    print('INFO: Test size: %d, Number of batches: %d' % (len(test_files), ntest))
    
    # Iterature through the files to be predicted
    start = time.time()
    for x_test, y_test, fname in generator_3dCoil(test_path, img_size, testing= True, shuffle_epoch=False):
        FNAME.append(fname[:-7])
        recon = model.predict(x_test, batch_size = 1) #Recon Image   

        im_ch1= x_test[0,:,:,0]
        im_ch2= x_test[0,:,:,1]
        im_ch3= x_test[0,:,:,2]
        im_pred = recon[0,:,:,0]
        im_gt = y_test[0,:,:,0]  
        
        # Calculate Peak Signal Intensities
        ch1_peak.append(np.max(im_ch1))
        ref_peak.append(np.max(im_gt))
        pred_peak.append(np.max(im_pred))
        
        # Calculate Noise levels
        N = 20
        ch1_noise.append(np.std(im_ch1[-N:,-N:], axis=None))
        ref_noise.append(np.std(im_gt[-N:,-N:], axis=None))
        pred_noise.append(np.std(im_pred[-N:,-N:], axis=None))

        
        #print(fname[:-7], ' Noise levels: \t', np.std(im_ch1[-N:,-N:], axis=None),'\t', np.std(im_ch2[-N:,-N:], axis=None),'\t', np.std(im_ch3[-N:,-N:], axis=None), '\t', np.std(im_pred[-N:,-N:], axis=None),'\t',np.std(im_gt[-N:,-N:]))
        
        #Calculate SNR images
        im_ch1 = im_ch1/np.std(im_ch1[-N:, -N:], axis=None)
        im_ch2 = im_ch2/np.std(im_ch2[-N:, -N:], axis=None)
        im_ch3 = im_ch3/np.std(im_ch3[-N:, -N:], axis=None)
        im_pred = im_pred/np.std(im_pred[-N:, -N:], axis=None)
        im_gt = im_gt/np.std(im_gt[-N:, -N:], axis=None)

        #Calculate Max of SNR for each image
        ch1_SNRmax.append(np.max(im_ch1))
        ref_SNRmax.append(np.max(im_gt))
        pred_SNRmax.append(np.max(im_pred))

        #Calculate Average of SNR for each image
        ch1_SNRave.append(np.mean(im_ch1))
        ref_SNRave.append(np.mean(im_gt))
        pred_SNRave.append(np.mean(im_pred))   
        
        vmax = np.max(im_gt)
        # Plot
        fig = plt.figure(num=None, figsize=[5, 2.0], dpi=320.0, frameon=False)
        fig.suptitle ('Ch1_M=%.1f, Pred_M=%.1f, GT_M=%.1f \n%s'%(np.max(im_ch1), np.max(im_pred), np.max(im_gt), fname[0:32]))
      
        ax0 = fig.add_subplot(1,5,1, aspect='equal') 
        im=ax0.imshow(im_ch1,  vmin=0, vmax=vmax, cmap='gray')
        ax0.title.set_text('Ch-1')
        plt.xticks([])
        plt.yticks([])
        
        ax1 = fig.add_subplot(1,5,2, aspect='equal')
        im=ax1.imshow(im_ch2, vmin=0, vmax=vmax, cmap='gray')
        ax1.title.set_text('Ch-2')
        plt.xticks([])
        plt.yticks([])
        
        ax1 = fig.add_subplot(1,5,3, aspect='equal')
        im=ax1.imshow(im_ch3,  vmin=0, vmax=vmax, cmap='gray')
        ax1.title.set_text('Ch-3')
        plt.xticks([])
        plt.yticks([])
        
        ax1 = fig.add_subplot(1,5,4, aspect='equal')
        im=ax1.imshow(im_pred, vmin=0, vmax=vmax, cmap='viridis')
        ax1.title.set_text('Pred')
        plt.xticks([])
        plt.yticks([])

        ax1 = fig.add_subplot(1,5,5, aspect='equal')
        im=ax1.imshow(im_gt,  vmin=0, vmax=vmax, cmap='viridis')
        ax1.title.set_text('SoS')
        plt.xticks([])
        plt.yticks([])

        plt.subplots_adjust(top=0.55, bottom=0.03, left=0.03, hspace=0.02, wspace=0.02)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="8%", pad=-0.05)
        plt.colorbar(im, cax=cax, orientation="vertical")
        
        plt.savefig(test_result_path + '/'+ fname[:-7] + '.png')
        
        if img_cnt % 10 ==0:
            print ('Completed Amount=%.1f percent'%(100*img_cnt/ntest))
            elapsed = time.time()-start
            print ('Remaining time=%.1f seconds'%(elapsed/(img_cnt+1)*(ntest-img_cnt)))
            plt.show()
        
        plt.close()
            
        img_cnt += 1
        if img_cnt == ntest:
            break

    SAVE_PARAM = [ch1_peak, pred_peak, ref_peak, ch1_noise, pred_noise, ref_noise,
                  ch1_SNRmax, pred_SNRmax, ref_SNRmax,
                  ch1_SNRave, pred_SNRave, ref_SNRave]
    np.savetxt('Sub01_Trained_Sub0203Tested.txt', np.transpose(SAVE_PARAM), fmt='%.4f', delimiter='\t')
    with open('Sub01_Trained_Sub0203Tested.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(FNAME)
    
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    test_denoising(test_result_path, test_path, save_file, model_weights)


