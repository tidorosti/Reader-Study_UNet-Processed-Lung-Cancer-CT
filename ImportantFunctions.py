import os
import sys
import numpy as np
import math
import scipy.io
import random
import copy
import matplotlib.pyplot as plt
import time
#import astra
from datetime import datetime
from scipy.stats import wilcoxon
#import imshow_grid as ig
from skimage.metrics import structural_similarity as ssim

from IPython import display

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from tensorboard import summary
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Conv3DTranspose, Lambda, Reshape, Add, MaxPooling2D, UpSampling2D, Subtract, Activation
from tensorflow.keras.layers import Concatenate
import tensorflow.keras.backend as K
from IPython.display import clear_output



###################################################
#GENERATOR FUNCTIONS
###################################################

def generate_batches_residual(path, geometry, batch_size, angle_list):
    """
    Generator function to output batches of sparse data and labels (i.e. difference between gt and sparse data).
    
    Parameters:
    -----------
    path: path to dataset, string
    geometry: beam geometry (e.g. parallel), string
    batchsize: batchsize, int
    angle_list: subsampling done to dataset, list of all contributing number of angles

    Returns:
    --------
    output: batch of sparse artifacts, batch of full data
    """
    path_geometry = path + "/{}/".format(geometry)
    path_labels = path_geometry + "{}_angles".format(2048)
    
    while True:
        batch_labels = []
        batch_sparse = []
    
        for j in range(batch_size):
            
            angle=np.random.choice(angle_list)
            
            pat_nr = os.listdir(path_labels)[random.randrange(0, len(os.listdir(path_labels)))]
            while not pat_nr.startswith("D") or not pat_nr.startswith("A"):
                pat_nr=os.listdir(path_labels)[random.randrange(0, len(os.listdir(path_labels)))]
            slice_nr = os.listdir(path_labels+"/"+pat_nr)[random.randrange(0, len(os.listdir(path_labels+"/"+pat_nr)))]
            while not slice_nr.startswith("D") or not pat_nr.startswith("A"):
                slice_nr = os.listdir(path_labels+"/"+pat_nr)[random.randrange(0, len(os.listdir(path_labels+"/"+pat_nr)))]
                
            labels_path = path_labels + "/" + pat_nr + "/" + slice_nr
            sparse_path = path_geometry + "{}_angles".format(angle) + "/" + pat_nr + "/" + slice_nr

            data_gt = np.load(labels_path, mmap_mode='c') 
            data_sparse = np.load(sparse_path, mmap_mode='c')
            assert(not np.isinf(data_sparse).any())
                      
            data_gt_norm = clip_and_norm(data_gt.astype(np.float32))
            data_sparse_norm = clip_and_norm(data_sparse.astype(np.float32))
            
            data_labels= data_gt_norm-data_sparse_norm #residual images (streak artifacts) as labels instead of high resolution images
            
            batch_labels.append(data_labels.astype('float16'))
            batch_sparse.append(data_sparse_norm.astype('float16'))
        
        yield np.reshape(np.array(batch_sparse), (batch_size, 512, 512, 1)), \
            np.reshape(np.array(batch_labels), (batch_size, 512, 512, 1))

def get_all_test_data_residual(path, patient, geometry, num_angles, get_names=False):
    '''
    Function to get sparse test input data and test labels (sparse artifacts).
    
    Parameters:
    -----------
    path: path to dataset, string
    patient: patient ID/name, string
    geometry: beam geometry (e.g. parallel), string
    num_angles: number of projection view angles, int
    get_names: flag to return names, bool (default: False)
    
    Returns:
    --------
    output: array of sparse image data, array of corresponding image labels, list of slice names
    '''
       
    sparse_data=[]
    label_data=[]
    gt_img=[]
    slice_name=[]
    
    path_geometry=path + "/{}/".format(geometry)
    path_num_angles=path_geometry + "{}_angles/{}".format(num_angles, patient)
        
    
    num_pred_samples = len(os.listdir(path_num_angles))



    for nr in range(0, num_pred_samples, 4):

        label_path = path_geometry + "{}_angles/".format(2048) + "/" + os.listdir(path_pat)[nr]
        sparse_path = path_num_angles+ "/" + patient + "/" +os.listdir(path_num_angles)[nr]
                    
        sample_sp = np.load(sparse_path, mmap_mode='c') 
        sample_gt = np.load(label_path, mmap_mode='c')

        sample_gt_norm = clip_and_norm(sample_gt.astype(np.float32))
        sample_sp_norm = clip_and_norm(sample_sp.astype(np.float32))

        sample_label=sample_gt_norm-sample_sp_norm
        
        sparse_data.append(sample_sp_norm.astype('float16'))
        label_data.append(sample_label.astype('float16'))
        gt_img.append(sample_gt_norm.astype("float16"))

        if get_names:
            slice_name.append(os.listdir(path_num_angles)[nr][:-4])

    return np.reshape(np.array(sparse_data), (len(sparse_data), 512, 512, 1)), np.reshape(np.array(label_data), (len(label_data), 512, 512, 1)), np.reshape(np.array(gt_img), (len(gt_img), 512, 512, 1)), slice_name

def get_all_test_data_sev_angles_residual(path, patient, geometry, angle_list, get_names=False):
    '''
    Function to get sparse test input data and test labels (sparse artifacts), if input data consists of multiple angle views.
    
    Parameters:
    -----------
    path: path to dataset, string
    patient: patient ID/name, string
    geometry: beam geometry (e.g. parallel), string
    angle_list: list of projection view angles
    get_names: flag to return names, bool (default: False)
    
    Returns:
    --------
    output: array of sparse image data, array of corresponding image labels, list of slice names
    '''
       
    sparse_ct=[]
    label_data=[]
    slice_name=[]
    gt_ct=[]
    
    path_geometry=path + "/{}/".format(geometry)
    path_gt=path_geometry + "{}_angles/{}".format(2048, patient)
    
    num_pred_samples = len(os.listdir(path_gt))
    
    for nr in range(0, num_pred_samples, 4):

        angle=np.random.choice(angle_list)

        gt_path = path_gt + "/" + os.listdir(path_gt)[nr]
        sparse_path = path_geometry + "{}_angles/{}".format(angle, patient) + "/" + os.listdir(path_gt)[nr]

        sample_sp = np.load(sparse_path, mmap_mode='c') 
        sample_gt = np.load(gt_path, mmap_mode='c')
        
        sample_gt_norm = clip_and_norm(sample_gt.astype(np.float32))
        sample_sp_norm = clip_and_norm(sample_sp.astype(np.float32))
        
        sample_label=sample_gt_norm-sample_sp_norm
        
        sparse_ct.append(sample_sp_norm.astype('float16'))
        label_data.append(sample_label.astype('float16'))
        gt_ct.append(sample_gt_norm.astype("float16"))

        if get_names:
            slice_name.append(os.listdir(path_gt)[nr][:-4])
    
    print(np.array(sparse_ct).shape, np.array(label_data).shape, np.array(gt_ct).shape)
    return np.reshape(np.array(sparse_ct), (len(sparse_ct), 512, 512, 1)), np.reshape(np.array(label_data), (len(label_data), 512, 512, 1)), np.reshape(np.array(gt_ct), (len(gt_ct), 512, 512, 1)), slice_name

def get_number_of_steps(path, geometry, batchsize):
    '''
    Get number of steps per epoch based on batchsize.
    
    Parameters:
    ----------
    path: path to the data of dataset, string (e.g. train or validation set)
    geometry: beam geometry, string
    batchsize: batch size, int
    
    Returns:
    --------
    output: number of steps per epoch, int
    '''
    tempsize=0
    for patient in os.listdir(path + "/{}/2048_angles".format(geometry)):
        num_per_patient=len(os.listdir(path + "/{}/2048_angles/{}".format(geometry, patient)))
        if patient.startswith("D"):
            print(path + "/{}/2048_angles/{}".format(geometry, patient), len(os.listdir(path + "/{}/2048_angles/{}".format(geometry, patient))))
            tempsize=tempsize+num_per_patient
    return int(math.floor(tempsize/batchsize))


##################################################################
#TRAINING FUNCTIONS
##################################################################	

# adapted from: https://github.com/jaejun-yoo/framing-unet/tree/master
def make_model(tag, init_lr, N=512):
    ''' 
    Create the U-Net model and compiles it with MSE loss and Adam optimizer.
    
    Parameters:
    -----------
    tag: the unet tag (Unet or dualUnet), string
    init_lr: initial learning rate
    N: input image size, int (default: 512)
    
    Returns:
    --------
    output: the created model, keras model
    '''
    
    if tag == "Unet":
        print(tag)
        # Build model - Unet
        input_shape = (N,N,1)
        model_input = Input(shape=input_shape)
        conv1 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block1_conv1')(model_input)
        conv1 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block1_conv2')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block2_conv1')(pool1)
        conv2 = Conv2D(128, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block2_conv2')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block3_conv1')(pool2)
        conv3 = Conv2D(256, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block3_conv2')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block4_conv1')(pool3)
        conv4 = Conv2D(512, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block4_conv2')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block5_conv1')(pool4)
        conv5 = Conv2D(512, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block5_conv2')(conv5)
        up1 = UpSampling2D(size = (2,2))(conv5)
        merge1 = Concatenate()([conv4,up1])
        conv6 = Conv2D(512, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block6_conv1')(merge1)
        conv6 = Conv2D(256, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block6_conv2')(conv6)
        up2 = UpSampling2D(size = (2,2))(conv6)
        merge2 = Concatenate()([conv3,up2])

        conv7 = Conv2D(256, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block7_conv1')(merge2)
        conv7 = Conv2D(128, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block7_conv2')(conv7)
        up3 = UpSampling2D(size = (2,2))(conv7)
        merge3 = Concatenate()([conv2,up3])

        conv8 = Conv2D(128, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block8_conv1')(merge3)
        conv8 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block8_conv2')(conv8)
        up4 = UpSampling2D(size = (2,2))(conv8)
        merge4 = Concatenate()([conv1,up4])

        conv9 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block9_conv1')(merge4)
        conv9 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block9_conv2')(conv9)

        model_output = Conv2D(1, (1, 1), 
                   use_bias=False, padding="same",activation="linear",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block9_conv3')(conv9)
    
    if tag == "dualUnet":
        print(tag)
        # Build model - dual Unet
        input_shape = (N,N,1)
        model_input = Input(shape=input_shape)
        conv1 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block1_conv1')(model_input)
        conv1 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block1_conv2')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block2_conv1')(pool1)
        conv2 = Conv2D(128, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block2_conv2')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block3_conv1')(pool2)
        conv3 = Conv2D(256, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block3_conv2')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block4_conv1')(pool3)
        conv4 = Conv2D(512, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block4_conv2')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block5_conv1')(pool4)
        conv5 = Conv2D(512, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block5_conv2')(conv5)
        resid1 = Subtract()([conv5,pool4])
        up1 = UpSampling2D(size = (2,2))(resid1)
        merge1 = Concatenate()([conv4,up1])
        conv6 = Conv2D(512, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block6_conv1')(merge1)
        conv6 = Conv2D(256, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block6_conv2')(conv6)
        resid2 = Subtract()([conv6,pool3])
        up2 = UpSampling2D(size = (2,2))(resid2)
        merge2 = Concatenate()([conv3,up2])

        conv7 = Conv2D(256, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block7_conv1')(merge2)
        conv7 = Conv2D(128, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block7_conv2')(conv7)
        resid3 = Subtract()([conv7,pool2])
        up3 = UpSampling2D(size = (2,2))(resid3)
        merge3 = Concatenate()([conv2,up3])

        conv8 = Conv2D(128, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block8_conv1')(merge3)
        conv8 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block8_conv2')(conv8)
        resid4 = Subtract()([conv8,pool1])
        up4 = UpSampling2D(size = (2,2))(resid4)
        merge4 = Concatenate()([conv1,up4])

        conv9 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block9_conv1')(merge4)
        conv9 = Conv2D(64, (3, 3), 
                   use_bias=False, padding="same",activation="relu",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block9_conv2')(conv9)

        model_output = Conv2D(1, (1, 1), 
                   use_bias=False, padding="same",activation="linear",
                   strides=1,kernel_initializer='glorot_uniform',
                   name='block9_conv3')(conv9)
    
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr) , metrics=['acc', psnr_tensor])
    return model


def make_or_restore_current_model(full_checkpoint_path, tag, init_lr, img_dim):
    ''' 
    Either restore the latest model, or create a fresh one if there is no checkpoint available.
    
    Parameters:
    ----------
    full_checkpoint_path: path to the saved model checkpoints, string
    tag: the unet tag (Unet or dualUnet), string
    init_lr: initial learning rate
    img_dim: input image size, int
    
    Returns:
    --------
    output: the created or restored model, keras model
    '''
    
    checkpoints = [full_checkpoint_path + '/' + name for name in os.listdir(full_checkpoint_path)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        model=load_model(latest_checkpoint, custom_objects={"psnr_tensor": psnr_tensor})
        model.compile(loss=model.loss, optimizer=model.optimizer, metrics=['acc', psnr_tensor])
        return model
    print('Creating a new model')
    return make_model(tag, init_lr, img_dim)



def restore_model_from_epoch(full_checkpoint_path, wanted_epoch):
    ''' 
    Either restore the latest model from the given epoch, or create a fresh one if there is no checkpoint available.
    
    Parameters:
    ----------
    full_checkpoint_path: path to the saved model checkpoints, string
    wanted_epoch: epoch number, int
    
    Returns:
    --------
    output: the created or restored model, keras model
    '''
    
    wanted_ckpt = full_checkpoint_path + '/' + str(wanted_epoch)
    print('Restoring from', wanted_ckpt)
    model=load_model(wanted_ckpt, custom_objects={"psnr_tensor": psnr_tensor})
    model.compile(loss=model.loss, optimizer=model.optimizer, metrics=['acc', psnr_tensor])
    return model


def set_new_lr(model, new_lr):
    '''
    Setting a new learning rate.
    
    Parameters:
    ----------
    model: the current mode, keras model
    new_lr: the new value to update the learning rate to
    
    Returns:
    --------
    only prints the old and the new learning rate values
    '''
    
    print("Old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, new_lr)
    print("New learning rate: {}".format(K.get_value(model.optimizer.lr)))


def scheduler(epoch, lr):
    '''
    Learning rate scheduler.
    
    Parameters:
    ----------
    epoch: epoch number, int
    lr: learning rate, float
    
    Returns:
    --------
    output: learning rate
    '''
    if epoch < 1:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
lr_scheduler = LearningRateScheduler(scheduler, verbose = 1)


###############################################################
# METRIC FUNCTIONS FOR TRAINING AND EVALUATION
###############################################################


def psnr_tensor(y_true,y_pred):      
    '''
    Peak signal to noise ratio (PSNR) calculator for tensor input data.
    
    Parameters:
    -----------
    y_true: ground truth label image, tensor
    y_pred: predicted image, tensor
    
    Returns:
    --------
    ouput: PSNR value, float
    '''

    y_pred_norm = (y_pred - K.min(y_pred))/(K.max(y_pred)-K.min(y_pred))
    y_true_norm = (y_true - K.min(y_true))/(K.max(y_true)-K.min(y_true))
    return 10.0 * (K.log(1.0 )-K.log((K.mean(K.square(y_pred_norm - y_true_norm))))) / K.log(10.0)

def psnr_npa(gt, pred):             
    '''
    Peak signal to noise ratio (PSNR) calculator for numpy array input data.
    
    Parameters:
    -----------
    gt: ground truth label image, nparray
    pred: predicted image, nparray
    
    Returns:
    --------
    ouput: PSNR value, float
    '''
    return 10.0 * (K.log(1.0 )-K.log((K.mean(K.square(pred - gt))))) / K.log(10.0)

def mean_psnr(preds, gt):
    '''
    Mean peak signal to noise ratio (PSNR) calculator for numpy array input data.
    
    Parameters:
    -----------
    gt: ground truth label image, nparray
    preds: predicted image, nparray
    
    Returns:
    --------
    ouput: mean PSNR value, float
    '''
    psnr_list = []
    for i in range(0,len(gt)):
        psnr_list.append(psnr_npa(gt.squeeze()[i], preds.squeeze()[i]))
    psnr_mean=np.mean(np.array(psnr_list))
    return psnr_mean

def mse(im_true, im_pred):
    '''
    Mean squared error (MSE) calculator.
    
    Parameters:
    -----------
    im_true: ground truth label image, nparray
    im_pred: predicted image, nparray
    
    Returns:
    --------
    ouput: MSE value, float
    '''
    mserr = np.mean( (im_true - im_pred) ** 2 )
    return mserr

def mean_mse(preds, gt):
    '''
    Mean mean squared error (MSE) calculator.
    
    Parameters:
    -----------
    gt: ground truth label image, nparray
    preds: predicted image, nparray
    
    Returns:
    --------
    ouput: mean MSE value, float
    '''
    mse_list = []
    for i in range(0,len(gt)):
        mse_list.append(mse(gt.squeeze()[i], preds.squeeze()[i]))
    mse_mean=np.mean(np.array(mse_list))
    return mse_mean

def mean_ssim(preds, gt):
    '''
    Mean structural similarity index measure (SSIM) calculator.
    
    Parameters:
    -----------
    gt: ground truth label image, nparray
    preds: predicted image, nparray
    
    Returns:
    --------
    ouput: mean SSIM value, float
    '''
    ssim_list = []
    for i in range(0,len(gt)):
        ssim_list.append(ssim(gt.squeeze()[i], preds.squeeze()[i], data_range = gt.squeeze()[i].max() - gt.squeeze()[i].min()))
    ssim_mean=np.mean(np.array(ssim_list))
    return ssim_mean

################################################################
#EVALUATION FUNCTIONS FOR MODEL TESTING
################################################################
def get_anglenames(angle_list):
    '''
    Given a list of projection view angle, returns the angle name.
    '''
    for i in range(len(angle_list)):
        angle_list[i]=str(angle_list[i])
    name="".join(angle_list)
    return(name)


def get_evaluation_residual(stand_val, spec_run, return_images=False, return_values=True, diff_testdata=[False, []], LungDict=None):
    '''
    Evaluating the model, given the test data, based on mean PSNR, SSIM, and MSE values.
    
    Parameters:
    ----------
    stand_val: a data class holding the model-relevant parameters
    spec_run: a data class holding the model-relevant parameters for a specific run
    return_images: flag to return input, and predicted images, bool (default: False)
    return_values: flag to return evaluation metric values, bool (default: True)
    diff_testdata: a list of a flag and list of projection view angles, list[bool,list] (flag default: False)
    LungDict: a dictionary to select only CT slices corresponding to lung region (default: None)
    
    Returns:
    -------
    outputs: input test images, predicted images, evluation metrics on patient level
    '''
    
    model_path = "{}/{}/{}_{}_bs{}_lr{}_ep{}/{}".format(stand_val.checkpoint_dir, stand_val.geometry, stand_val.net_type, get_anglenames(spec_run.angle_list), stand_val.batch_size, stand_val.learning_rate, stand_val.epochs, spec_run.chosen_date)
    model = restore_model_from_epoch(model_path, spec_run.wanted_epoch)
    
    predct_list=[]
    predart_list=[]
    gtart_list=[]
    gtct_list=[]
    sparsect_list=[]
    
    psnrct_list=[]
    msect_list=[]
    ssimct_list=[]
    
    psnrart_list=[]
    mseart_list=[]
    ssimart_list=[]
    
    all_names=[]
    
    for patient in os.listdir("{}/{}/{}_angles".format(stand_val.test_path, stand_val.geometry, 2048)):
        print(patient)
        if patient.startswith("D") or patient.startswith("A") or patient.startswith("I"):
            if diff_testdata[0]:
                sparse_ct, gtart, gtct, name_list = get_all_test_data_sev_angles_residual(stand_val.test_path, patient, stand_val.geometry, diff_testdata[1], get_names=True)
            else:
                sparse_ct, gtart, gtct, name_list = get_all_test_data_sev_angles_residual(stand_val.test_path, patient, stand_val.geometry, spec_run.angle_list, get_names=True)

            print(len(sparse_ct), len(gtart), len(name_list))
            if not LungDict==None:
                if patient in LungDict.keys():
                    mask=np.ones(len(name_list), dtype=bool)
                    print(LungDict[patient])
                    indizes=range(LungDict[patient][0], LungDict[patient][1]+1)
                    for i in range(len(sparse_ct)):
                        #print(name_list[i].split("_")[-1])
                        if not int(name_list[i].split("_")[-1]) in indizes:
                            mask[i]=False
                            name_list[i]=""
                    sparse_ct=sparse_ct[mask, ...]
                    gtart=gtart[mask, ...]
                    gtct=gtct[mask, ...]

                    while '' in name_list:
                        name_list.remove('')
                else:
                    print("patient %s not as key in LungDict"%patient)
            print(len(sparse_ct), len(gtart), len(name_list))         

            #calculate prediction for input based on model
            predart = model.predict(sparse_ct)
            
            #calculate evaluation values
            psnr_mean_art = mean_psnr(predart, gtart)
            mse_mean_art = mean_mse(predart, gtart)
            ssim_mean_art = mean_ssim(predart, gtart)

            predart_list.append(predart)
            gtart_list.append(gtart)
            sparsect_list.append(sparse_ct)

            psnrart_list.append(psnr_mean_art)
            mseart_list.append(mse_mean_art)
            ssimart_list.append(ssim_mean_art)
            
            
            predct=sparse_ct + predart
            
            psnr_mean_ct = mean_psnr(predct, gtct)
            mse_mean_ct = mean_mse(predct, gtct)
            ssim_mean_ct = mean_ssim(predct, gtct)

            predct_list.append(predct)
            gtct_list.append(gtct)


            psnrct_list.append(psnr_mean_ct)
            msect_list.append(mse_mean_ct)
            ssimct_list.append(ssim_mean_ct)

            all_names.append(name_list)

    if return_images and not return_values:
        return predart_list, sparsect_list, gtart_list, predct_list, gtct_list, all_names
    elif return_values and not return_images:
        return np.array(psnrart_list), np.array(mseart_list), np.array(ssimart_list), np.array(psnrct_list), np.array(msect_list), np.array(ssimct_list)
    elif return_images and return_values:
        return [predart_list, sparsect_list, gtart_list, all_names], [np.array(psnrart_list), np.array(mseart_list), np.array(ssimart_list)], [predct_list, gtct_list], [np.array(psnrct_list), np.array(msect_list), np.array(ssimct_list)]
            
########################################
#EVALUATION FUNCTIONS FOR READER STUDY
########################################   

def getLabelMeanPerAngle(l, angles=[16, 32, 64, 128, 256]):
    '''
    Given a label list, calculates the mean per angle.
    
    Parameters:
    -----------
    l: label list
    angles: the projection angles the data was collected over, list of int (defualt: [16, 32, 64, 128, 256])
    
    Returns:
    --------
    output: four lists containing two lists each corresponding to the label data, for sparse and U-Net methods
    '''
    
    l_mean = [0] * len(angles)
    for i in range(len(angles)):
        l_mean[i]=np.nanmean(np.array(l[i]))
    return l_mean
    

def getLabelsFromDf(df, angles=[16, 32, 64, 128, 256]):
    '''
    Extracts the data for each label (quality, confidence, artifact, dice), for sparse, and U-Net method, given the raw dataframe of reader study results.
    
    Parameters:
    -----------
    df: dataframe of results to be analyzed, pandas dataframe
    angles: the projection angles the data was collected over, list of int (defualt: [16, 32, 64, 128, 256])
    
    Returns:
    --------
    output: four lists containing two lists each corresponding to the label data, for sparse and U-Net methods
    '''
    
    quality_sparse, quality_unet = [[], [], [], [], []], [[], [], [], [], []]#[[]]* len(angles), [[]]* len(angles) #
    confidence_sparse, confidence_unet = [[], [], [], [], []], [[], [], [], [], []]#[[]]* len(angles), [[]]* len(angles) #
    artifact_sparse, artifact_unet = [[], [], [], [], []], [[], [], [], [], []]#[[]]* len(angles), [[]]* len(angles) #
    dice_sparse, dice_unet = [[], [], [], [], []], [[], [], [], [], []]#[[]]* len(angles), [[]]* len(angles) #

    for idx in range(len(df["patientid"])):
        if df["method"][idx]=="sparse":
            for ind, angle in enumerate(angles):
                if angle==df["angles"][idx]  and df["len_gtmask"][idx]>0 and df["len_rmask"][idx]>0:
                    dice_sparse[ind].append(df["dice"][idx])
                if angle==df["angles"][idx]:
                    quality_sparse[ind].append(df["quality"][idx])
                    confidence_sparse[ind].append(df["confidence"][idx])
                    artifact_sparse[ind].append(df["artifacts"][idx])
        elif df["method"][idx]=="predct":
            for ind, angle in enumerate(angles):
                if angle==df["angles"][idx] and df["len_gtmask"][idx]>0 and  df["len_rmask"][idx]>0: 
                    dice_unet[ind].append(df["dice"][idx])
                if angle==df["angles"][idx]:
                    quality_unet[ind].append(df["quality"][idx])
                    confidence_unet[ind].append(df["confidence"][idx])
                    artifact_unet[ind].append(df["artifacts"][idx])
    
    return [quality_sparse, quality_unet], [confidence_sparse, confidence_unet], [artifact_sparse, artifact_unet], [dice_sparse, dice_unet]

def getLabelsFromDf_readerLevel(df, angles=[16, 32, 64, 128, 256]):
    '''
    Extracts the reader-level (3 readers) data for each label (quality, confidence, artifact, dice), for sparse, and U-Net method, given the raw dataframe of reader study results.
    
    Parameters:
    -----------
    df: dataframe of results to be analyzed, pandas dataframe
    angles: the projection angles the data was collected over, list of int (defualt: [16, 32, 64, 128, 256])
    
    Returns:
    --------
    output: four lists containing two lists each, contatining three lists each corresponding to the label data, for sparse and U-Net methods, for the three readers
    '''
    
    quality_sparse_r1, quality_sparse_r2, quality_sparse_r3 = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []] #[[]]* len(angles), [[]]* len(angles), [[]]* len(angles)
    quality_unet_r1, quality_unet_r2, quality_unet_r3 = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []] #[[]]* len(angles), [[]]* len(angles), [[]]* len(angles)

    confidence_sparse_r1, confidence_sparse_r2, confidence_sparse_r3 = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []] #[[]]* len(angles), [[]]* len(angles), [[]]* len(angles)
    confidence_unet_r1, confidence_unet_r2, confidence_unet_r3 = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []] #[[]]* len(angles), [[]]* len(angles), [[]]* len(angles)

    artifact_sparse_r1, artifact_sparse_r2, artifact_sparse_r3 = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []] #[[]]* len(angles), [[]]* len(angles), [[]]* len(angles)
    artifact_unet_r1, artifact_unet_r2, artifact_unet_r3 = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []] #[[]]* len(angles), [[]]* len(angles), [[]]* len(angles)

    dice_sparse_r1, dice_sparse_r2, dice_sparse_r3 = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []] #[[]]* len(angles), [[]]* len(angles), [[]]* len(angles)
    dice_unet_r1, dice_unet_r2, dice_unet_r3 = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []] #[[]]* len(angles), [[]]* len(angles), [[]]* len(angles)

    for idx in range(len(df["patientid"])):
        if df["method"][idx]=="sparse":
            for ind, angle in enumerate(angles):
                if angle==df["angles"][idx] and df["len_gtmask"][idx]>0 and df["len_rmask"][idx]>0 :
                    if df["reader"][idx]=='r1':
                        dice_sparse_r1[ind].append(df["dice"][idx])
                    elif df["reader"][idx]=='r2':
                        dice_sparse_r2[ind].append(df["dice"][idx])
                    elif df["reader"][idx]=='r3':
                        dice_sparse_r3[ind].append(df["dice"][idx])
                if angle==df["angles"][idx]:
                    if df["reader"][idx]=='r1':
                        quality_sparse_r1[ind].append(df["quality"][idx])
                        confidence_sparse_r1[ind].append(df["confidence"][idx])
                        artifact_sparse_r1[ind].append(df["artifacts"][idx])
                    elif df["reader"][idx]=='r2':
                        quality_sparse_r2[ind].append(df["quality"][idx])
                        confidence_sparse_r2[ind].append(df["confidence"][idx])
                        artifact_sparse_r2[ind].append(df["artifacts"][idx])
                    elif df["reader"][idx]=='r3':
                        quality_sparse_r3[ind].append(df["quality"][idx])
                        confidence_sparse_r3[ind].append(df["confidence"][idx])
                        artifact_sparse_r3[ind].append(df["artifacts"][idx])

        elif df["method"][idx]=="predct":
            for ind, angle in enumerate(angles):
                if angle==df["angles"][idx] and df["len_gtmask"][idx]>0 and df["len_rmask"][idx]>0 :
                    if df["reader"][idx]=='r1':
                        dice_unet_r1[ind].append(df["dice"][idx])
                    elif df["reader"][idx]=='r2':
                        dice_unet_r2[ind].append(df["dice"][idx])
                    elif df["reader"][idx]=='r3':
                        dice_unet_r3[ind].append(df["dice"][idx])
                if angle==df["angles"][idx]:
                    if df["reader"][idx]=='r1':
                        quality_unet_r1[ind].append(df["quality"][idx])
                        confidence_unet_r1[ind].append(df["confidence"][idx])
                        artifact_unet_r1[ind].append(df["artifacts"][idx])
                    elif df["reader"][idx]=='r2':
                        quality_unet_r2[ind].append(df["quality"][idx])
                        confidence_unet_r2[ind].append(df["confidence"][idx])
                        artifact_unet_r2[ind].append(df["artifacts"][idx])
                    elif df["reader"][idx]=='r3':
                        quality_unet_r3[ind].append(df["quality"][idx])
                        confidence_unet_r3[ind].append(df["confidence"][idx])
                        artifact_unet_r3[ind].append(df["artifacts"][idx])
    
    return [[quality_sparse_r1, quality_sparse_r2, quality_sparse_r3],[quality_unet_r1, quality_unet_r2, quality_unet_r3]],[[confidence_sparse_r1, confidence_sparse_r2, confidence_sparse_r3],[confidence_unet_r1, confidence_unet_r2, confidence_unet_r3]], [[artifact_sparse_r1, artifact_sparse_r2, artifact_sparse_r3],[artifact_unet_r1, artifact_unet_r2, artifact_unet_r3]], [[dice_sparse_r1, dice_sparse_r2, dice_sparse_r3],[dice_unet_r1, dice_unet_r2, dice_unet_r3]]

def clustered_signed_rank_wilcoxon(x, y, clusters):
    """
    Perform a clustered signed-rank Wilcoxon non-parametric test.

    Parameters:
    -----------
    x: A list or numpy array representing the first set of paired observations
    y: A list or numpy array representing the second set of paired observations
    clusters: A list or numpy array specifying the cluster assignments for each pair

    Returns:
    --------
    statistic: The test statistic
    p_value: The p-value for the test
    """
    unique_clusters = np.unique(clusters)
    signed_rank_statistic_p = 0.0
    signed_rank_statistic_z = 0.0
    

    for cluster_id in unique_clusters:
        cluster_x = [x[i] for i in range(len(x)) if clusters[i] == cluster_id]
        cluster_y = [y[i] for i in range(len(y)) if clusters[i] == cluster_id]
        

        if len(cluster_x) != len(cluster_y):
            raise ValueError("The number of observations in each cluster must be equal.")

        z, signed_ranks = wilcoxon(cluster_x, cluster_y)
        
        signed_rank_statistic_p += np.sum(signed_ranks)
        signed_rank_statistic_z += np.sum(z)

    return signed_rank_statistic_z, signed_rank_statistic_p

def plot_subplot(ax, data_sparse, data_unet, title, ylabel, ylim=None, colors_sparse=None, colors_unet=None):
    '''
    Helper function for plotting results of reader study.
    '''
    if colors_sparse is None:
        colors_sparse = ["indianred", "khaki", "mediumseagreen", "skyblue", "mediumpurple"]
    if colors_unet is None:
        colors_unet = ["brown", "gold", "green","deepskyblue","rebeccapurple"]
    
    for i in range(5):
        ax.bar(x[i] - 0.5 * width, data_sparse[4 - i], width, color=colors_sparse[i])
        ax.bar(x[i] + 0.5 * width, data_unet[4 - i], width, color=colors_unet[i])
    
    ax.set_ylabel(ylabel, fontsize="x-large")
    ax.set_title(title, fontsize="x-large")
    ax.set_xticks(x, ["16", "32", "64", "128", "256"], fontsize="x-large")
    if ylim:
        ax.set_ylim(0, ylim)
    #ax.set_yticks(fontsize="x-large")
    ax.grid(alpha=0)
    
########################################
#FUNCTIONS TO WORK ON IMAGES
########################################
def clip_and_norm(image):
    ''' Clip an image to lung window and normalize to 0-1. '''
    return (np.clip(image, -1450., 250.) + 1450.)/ 1700.

def denorm(image):
    ''' Inverse clip_and_norm. '''
    return image*1700-1450

def circ_in_image(img, coord, diameter):
    ''' Create a circle mask on an image, given the coordinates and diameter of the ROI. '''
    mask=np.zeros(img.shape)

    x_coord=coord[0]
    y_coord=coord[1]
    z_coord=coord[2]
    circle =  ((np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))[0]-x_coord)**2 +(np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))[1]-y_coord)**2 <(diameter/2)**2) 
    mask=mask.squeeze() + circle.squeeze()

    roi=img*mask
    return mask[:,:, np.newaxis], roi

def scale_down(img, final_size):
    ''' Scale fown an image to a given final size. '''
    
    if img.shape[0]<final_size[0] or img.shape[1]<final_size[1]:
        print("new size is larger than old size, use other size")
    else:
        xfact=img.shape[0]//final_size[0]
        yfact=img.shape[1]//final_size[1]
        
        yind=0
        small_arr=np.zeros(final_size)
        
        while yind<final_size[1]:
            xind=0
            while xind<final_size[0]:
                
                if (xind+1)*xfact<img.shape[0] and (yind+1)*yfact<img.shape[1]:
                    subarr=img[xind*xfact:(xind+1)*xfact, yind*yfact:(yind+1)*yfact]
                    small_arr[xind, yind]=np.nanmean(subarr)
                elif (xind+1)*xfact>=img.shape[0] and (yind+1)*yfact<img.shape[1]:
                    subarr=img[xind*xfact:, yind*yfact:(yind+1)*yfact]
                    small_arr[xind, yind]=np.nanmean(subarr)

                elif (xind+1)*xfact<img.shape[0] and (yind+1)*yfact>=img.shape[1]:
                    subarr=img[xind*xfact:(xind+1)*xfact, yind*yfact:]
                    small_arr[xind, yind]=np.nanmean(subarr)
                elif (xind+1)*xfact>=img.shape[0] and (yind+1)*yfact>=img.shape[1]:
                    subarr=img[xind*xfact:, yind*yfact:]
                    small_arr[xind, yind]=np.nanmean(subarr)
                xind+=1
            yind=yind+1
            
    return small_arr


########################################
#FUNCTIONS TO PERFORM SPARSE SAMPLING
########################################


def get_sparse_data_slice(data_slice, angles, filter_type="ram-lak", geometry="parallel", beam_shape="flat", det_width=1.0, det_count=50, source_origin=570, origin_det=470, processing_unit="GPU"):
    
    '''
    Generate volume geometry an projection geometry as wanted.
    
    Parameters:
    ----------
    data_slice: the data to sparse sample, nparray
    angles: list of all projection angles
    filter_type: reconstruction filter (default: ram-lak)
    geometry: reconstruction geometry (default: parallel)
    beam_shape: shape of the source beam (default: flat)
    det_width, det_count: detector width (float) and count (int) (default: 1.0, 50)
    source_origin, origin_det: distances (int) of source to object and object to detector, respectively
    processing_unit: CPU or GPU, string (default: GPU)
    
    Returns:
    --------
    output: the sparse sampled input data
    
    Reference:
    ---------
    https://astra-toolbox.com/docs/index.html
    '''

    if processing_unit=="CPU":
        if geometry=="parallel":
            #2D volume with shape of slice:
            fp_vol_geom = astra.create_vol_geom(data_slice.shape) 
            #2D volume with shape (512, 512) to project to:
            fp_vol2_geom = astra.create_vol_geom((512, 512)) 
            #geometry behind final projection:
            fp_proj_geom = astra.create_proj_geom(geometry, det_width, det_count, angles) 
            
            if beam_shape=="flat":
                fp_projector_id = astra.create_projector("linear", fp_proj_geom, fp_vol_geom)
            elif beam_shape=="strip":
                fp_projector_id = astra.create_projector("strip", fp_proj_geom, fp_vol_geom)
        
        elif geometry=="fanflat":
            #2D volume with shape of slice:
            fp_vol_geom = astra.create_vol_geom(data_slice.shape) 
            #2D volume with shape (512, 512) to project to:
            fp_vol2_geom = astra.create_vol_geom((512, 512)) 
            #geometry behind final projection:
            fp_proj_geom = astra.create_proj_geom(geometry, det_width, det_count, angles, source_origin, origin_det) 
            
            if beam_shape == "flat":
                fp_projector_id = astra.create_projector("line_fanflat", fp_proj_geom, fp_vol_geom)
            elif beam_shape == "strip":
                fp_projector_id = astra.create_projector("strip_fanflat", fp_proj_geom, fp_vol_geom)
    else:
        if geometry=="parallel":
            #2D volume with shape of slice:
            fp_vol_geom = astra.create_vol_geom(data_slice.shape) 
            #2D volume with shape (512, 512) to project to:
            fp_vol2_geom = astra.create_vol_geom((512, 512)) 
            #geometry behind final projection:
            fp_proj_geom = astra.create_proj_geom(geometry, det_width, det_count, angles) 
            
        elif geometry=="fanflat":
            #2D volume with shape of slice:
            fp_vol_geom = astra.create_vol_geom(data_slice.shape) 
            #2D volume with shape (512, 512) to project to:
            fp_vol2_geom = astra.create_vol_geom((512, 512)) 
            #geometry behind final projection:
            fp_proj_geom = astra.create_proj_geom(geometry, det_width, det_count, angles, source_origin, origin_det) 
    
        fp_projector_id = astra.create_projector("cuda", fp_proj_geom, fp_vol_geom)
        
    # forward projection of data_slice to sinogram with projector fp_projector_id:
    my_sino_id, my_sino = astra.create_sino(data_slice, fp_projector_id) 
        
    #reconstruction of sparse projection:
    
    #creates volume ("-vol") data object following geometry in fp_vol_geom
    reco_id = astra.data2d.create("-vol", fp_vol2_geom, 0) 

    #configurations of algorithm
    config=astra.astra_dict("FBP_CUDA") #algorithm configuration struct for FBP CUDA algorithm
    config["ProjectionDataId"] = my_sino_id
    config["ReconstructionDataId"] = reco_id
    config["FilterType"] = "ram-lak" #"shepp-logan", "hamming", "cosine", ... others possible
    
    #create ("create") algorithm following configurations in config ->here: FBP algorithm:
    fbp_id = astra.algorithm.create(config) 
    astra.algorithm.run(fbp_id)
    
    #get reconstructed volume created by algorithm (identified via reco_id (config.ReconstructionDataId)):
    fpb_vol=astra.data2d.get(reco_id) 
    
    # Final clean up of GPU and RAM:
    astra.algorithm.delete(fbp_id)
    astra.data2d.delete(reco_id)
    astra.data2d.delete(my_sino_id)
    astra.projector.delete(fp_projector_id)
    
    return fpb_vol


###############################################################
#GENERAL FUNCTIONS THAT WILL BE NEEDED LATER
###############################################################
def readCSV(filename):
    '''Helper for reading CSV files'''
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


