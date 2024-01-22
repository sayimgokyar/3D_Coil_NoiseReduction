# Revised by Sayim Gokyar Jan, Jun, July 2021, Feb, Oct 2022, Jan 2024

from __future__ import print_function, division
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization

def unet_2d(input_size, stages, featurelength):
    # input_size = (dim1, dim2, ch) #There should be 3 channels for 3D coil!

    nfeatures = [2**feat*featurelength for feat in np.arange(stages)]
    depth = len(nfeatures)    
    conv_ptr = []
    inputs = Input(input_size) 
    pool = inputs

    for depth_cnt in range(depth):
        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(pool)
        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(conv)
        
        #conv = BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv_ptr.append(conv)

        if depth_cnt < depth-1:
            # If size of input is odd, only do a 3x3 max pool
            xres = conv.shape.as_list()[1]
            if (xres % 2 == 0):
                pooling_size = (2,2)
            elif (xres % 2 == 1):
                pooling_size = (3,3)

            pool = MaxPooling2D(pool_size=pooling_size)(conv)

    # step up convolutional layers
    for depth_cnt in range(depth-2,-1,-1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None

        # If size of input is odd, then do a 3x3 deconv  
        if (deconv_shape[1] % 2 == 0):
            unpooling_size = (2,2)
        elif (deconv_shape[1] % 2 == 1):
            unpooling_size = (3,3)

        #print('INFO: outputshape: %s' % (deconv_shape,))
        up = concatenate([Conv2DTranspose(nfeatures[depth_cnt],(3,3),
                          padding='same',
                          strides=unpooling_size)(conv),
                          conv_ptr[depth_cnt]], 
                          axis=3)

        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(up)
        conv = Conv2D(nfeatures[depth_cnt], (3,3), 
                      padding='same', 
                      activation='relu',
                      kernel_initializer='he_normal')(conv)

    recon = Conv2D(1, (1,1), padding='same', activation='relu')(conv)   
    model = Model(inputs=[inputs], outputs=[recon])  
    return model

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

