
#pipfrom __future__ import print_function, division
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from utils.generator_3dCoil import file_list_info, generator_3dCoil
from utils.models import unet_2d as Model
print(__doc__)
ModelName = Model.__name__

initial_lrate = 1e-4
lyrs=2 #number of Unet Layers
conv_filters=96
n_epochs = 100

# Training and validation data locations
train_path = './training'
valid_path = './validation'
train_batch_size = 1
valid_batch_size = 1

# Locations and names for saving training checkpoints
cp_save_path = './weights/'
cp_save_tag = ModelName +  '_3dCoil' + f'_lr{initial_lrate:06f}'

def train_3dCoil(train_path, valid_path, train_batch_size, valid_batch_size, 
                cp_save_path, cp_save_tag, n_epochs):

    # set image format to be (N, dim1, dim2, ch)
    K.set_image_data_format('channels_last')
    train_files, train_nbatches, img_size = file_list_info(train_path, train_batch_size)
    valid_files, valid_nbatches, img_size = file_list_info(valid_path, valid_batch_size)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lrate,
        decay_steps=10*len(train_files),
        decay_rate=0.95,
        staircase=True)
    
    # Print some useful debugging information
    print('INFO: Train size: %d, batch size: %d' % (len(train_files), train_batch_size))
    print('INFO: Valid size: %d, batch size: %d' % (len(valid_files), valid_batch_size))
    print('INFO: Image size: %s' % (img_size,))

    # create the unet model
    model = Model(img_size, lyrs, conv_filters)
    model.summary()
    
    # Load pre-trained model
    model_weights = None #'./checkpoints/u....h5'
    if model_weights is not None:
        model.load_weights(model_weights)
        print('weights loaded from a pretrained model')

    model.compile(optimizer=Adam(learning_rate=lr_schedule, beta_1=0.90, beta_2=0.999, epsilon=1e-06, weight_decay=0.1), loss='mean_squared_error', metrics=(['mean_absolute_error']))
    cp_cb   = ModelCheckpoint(cp_save_path + '/' + cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5', save_best_only=True)
    csv_logger = CSVLogger(cp_save_path + '/Logs.csv', append=True)

    EarlyStopping(
    monitor="val_loss",
    min_delta=0.01,
    patience=20,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
    callbacks_list = [cp_cb, csv_logger]
    
    # Start the training    
    model.fit(
        generator_3dCoil(train_path, img_size),
        batch_size=1,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=generator_3dCoil(valid_path, img_size),
        shuffle=True,
        initial_epoch=0,
        steps_per_epoch=len(train_files),
        validation_steps=1,
        validation_batch_size=1,
        validation_freq=1,
        max_queue_size=100,
        workers=1)

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'

    train_3dCoil(train_path, valid_path, train_batch_size, valid_batch_size,
                cp_save_path, cp_save_tag, n_epochs)