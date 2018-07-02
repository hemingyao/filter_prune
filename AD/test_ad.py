# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
import os, cv2, h5py
#to control the memory uesd by tensorflow, without this lines tensorflow will consume all gpus and all memory
import tensorflow as tf
#my own functions
from scipy.misc import imresize
import glob
from tensorflow.python.framework import ops
###################
import network
from flags import FLAGS, TRAIN_RANGE, VAL_RANGE, option

import tflearn_dev as tflearn
from data_flow import input_fn, IMG_SIZE

from utils import multig, prune, op_utils, train_ops
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1" 
tf_config=tf.ConfigProto() 
tf_config.gpu_options.allow_growth=True 
tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
sess=tf.Session(config=tf_config) 
#odel_path ='/home/spc/Dropbox/GDCNN_test/output/no_shiftSunny_model1_02-05_14.35.16/model/val_loss=.089.hdf5'
output_path = '/home/spc/Documents/Small_Dataset/LV/'
#model_path = '/home/spc/Dropbox/Filter_Prune/Logs/model3_add_loss_more_2_weight5_lr_0.001_wd_0.0_Jun_22_16_52'\
#            '/model/epoch_70.0_acc_-0.496-35000'
model_path = '/home/spc/Dropbox/Filter_Prune/Logs/model3_add_loss_more_2_weight5_lr_0.001_wd_0.0_Jun_23_17_16'\
            '/model/epoch_20.0_acc_0.709-10000'
TEST_RANGE = range(0,100,10)

#model_path ='/home/spc/Dropbox/GDCNN_test/output/weight=1LV_Sparse_1_02-04_17.13.45/model/val_loss=.009.hdf5'

#################################################################################   
def dice_loss(y_pred, y_true):
    y_true = y_true.astype(np.float)
    inter = y_pred*y_true
    union = y_pred+y_true
    return -1*np.sum(y_pred*y_true)/(np.sum(union-inter)+0.00001)

def Jaccard(y_true,y_pred):
    # 2 is just a scaling factor
    # add a number at end (20) to avoid dividing by 0
    union = y_pred+y_true
    inter = y_pred*y_true
    return -np.sum(inter)/(np.sum(union)-np.sum(inter)+0.00001)

def mean(lis):
    return float(np.mean(np.array(lis)))

def var(lis):
    # Calculate the variance of values in a list
    return float(np.var(np.array(lis)))

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def padwithones(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 1
    vector[-pad_width[1]:] = 1
    return vector
#################################################################################  
def load_model(path):
    print('-'*50)
    print('Loading model...')
    model = keras.models.load_model(path)#,custom_objects={'loss': sorenson_dice})
    print('-'*50)   
    return model

#################################################################################  
def draw_output(prediction, image, label, subject, name, dir_name):
    path = '/home/spc/Documents/NewPrediction/' + dir_name + subject +'/'

    if not os.path.isdir(path):
        os.mkdir(path)

    concat_img= np.hstack((image, label, prediction))*255

    cv2.imwrite(path + str(name) + '.png', concat_img)

    #cv2.imwrite(path + name + 'pred.png', (prediction/2)*255)
    #cv2.imwrite(path + str(name) + 'label.png', (label)*255)
    #cv2.imwrite(path + str(name) + 'img.png', image*255)
    #cv2.destroyAllWindows()
    return True

def seg_acc_op(predictions, labels):
    TP=np.sum(predictions*labels)
    TN=np.sum((1-predictions)*(1-labels))
    FP=np.sum(predictions*(1-labels))
    FN=np.sum((1-predictions)*labels)
    TO=np.sum(labels)
    TB=np.sum(1-labels)

    PA=(TP+TN)/(TP+FN+FP+TN)
    SE=TP/TO
    SP=TN/TB
    return SE, SP, PA


def calculate_ef(pred_1, pred_2, label_1, label_2, slice_sp):
    auto_i_1 = np.sum(pred_1)
    auto_i_2 = np.sum(pred_2)
    manu_i_1 = np.sum(label_1)
    manu_i_2 = np.sum(label_2)

    ef_auto = abs(auto_i_1 - auto_i_2)/max(auto_i_1, auto_i_2)*100
    ef_manu = abs(manu_i_1 - manu_i_2)/max(manu_i_1, manu_i_2)*100

    return ef_auto, ef_manu


def calculate_mass(prediction, label, slice_sp):
    auto_mass = np.sum(prediction[:,:,:,1])*slice_sp*0.001*1.05  #1.05 (g/cm3)
    manu_mass = np.sum(label[:,:,:,1])*slice_sp*0.001*1.05
    return auto_mass, manu_mass


def calculate_volume(prediction, label, slice_sp):
    auto_volume = np.sum(prediction[:,:,:,2])*slice_sp*0.001 #1.05 (g/cm3)
    manu_volume = np.sum(label[:,:,:,2])*slice_sp*0.001
    return auto_volume, manu_volume


def rescale(img_block, scale):
    new_block = np.zeros(img_block.shape)
    for i in range(img_block.shape[0]):
        img = img_block[i,:]

        h, w = 256, 256
        if len(img.shape)==2:
            img = np.lib.pad(img, 60, padwithzeros)
            output = imresize(img,scale)
            center_h = output.shape[0]//2
            center_w = output.shape[1]//2
            output = output[center_h-h//2: center_h+h//2, center_w-w//2: center_w+w//2]
        elif len(img.shape)==3:
            img = np.lib.pad(img, 60, padwithzeros)
            temp = img.shape[-1]
            img = img[:,:,60:temp-60]
            output = imresize(img,scale, mode='RGB')
            center_h = output.shape[0]//2
            center_w = output.shape[1]//2
            output = output[center_h-h//2: center_h+h//2, center_w-w//2: center_w+w//2,:]
        new_block[i,:] = output
    return new_block/255


def main(model,data_path,  output_path, test_range=None):
    h5f = h5py.File(data_path, 'r')
    datatype='val'
    #datatype = FLAGS.dataset_test
    dice_all_epi = []; dice_all_endo = []; dice_all_wall = []; 
    
    #sn_list = []; sp_list = []; acc_list = []; all_dice = []
    
    #output_all = open(output_path+'/all_patients.txt', 'a')
    #output_sum = open(output_path+'/summary.txt', 'a')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    log_f = open(os.path.join(output_path, 'result_info'), 'a')
    for dataset in ['val']:
    #for dataset in ['train']:
        # Find patient ID in each dataset
        if datatype=='Sunny':
            subjects = list(h5f[dataset+'/location/'].keys())
        else:
            subjects = list(h5f['location/'].keys())
            test_range = list(test_range)
            subjects = [subjects[i] for i in test_range]
            print(subjects)

        # Read MRI slices for each patient
        sess = tf.Session(config=tf_config)
        save_root = FLAGS.save_root_for_prediction
        with sess.as_default():
            tflearn.is_training(False, session=sess)
            batch_data = tf.placeholder(tf.float32, shape=[1, 256,256,1], name='batch_data')
            batch_label = tf.placeholder(tf.float32, shape=[1, 256,256,3], name='batch_label')

            with tf.variable_scope(FLAGS.net_name):
                logits = getattr(network, FLAGS.net_name)(inputs=batch_data, 
                    prob_fc=1, prob_conv=1, wd=0, wd_scale=0, 
                    training_phase=False)
                logits = logits[0]

            print(logits.shape)

            label = batch_label[:,:,:,1:3]
            label = tf.greater(label, tf.ones(label.shape)*0.5)
            label = tf.cast(label, tf.float32)
            #label = tf.cast(tf.argmax(self.label_pl[:,:,:,0:2],3), tf.float32)
            #label = tf.expand_dims(label, -1)

            al_true = getattr(network, 'adversatial_2')(batch_data, label,reuse=False)
            al_true = tf.nn.softmax(al_true)

            y_pred = tf.nn.softmax(logits[0])
            y_pred = tf.reshape(y_pred, [1,256,256,3])
            maxvalue = tf.reduce_max(y_pred, axis=-1)
            print(y_pred.shape)
            y_pred = tf.equal(y_pred, tf.stack([maxvalue,maxvalue,maxvalue], axis=-1))
            pred = tf.cast(y_pred[:,:,:,1:], tf.float32)

            #print(pred)
            al_false = getattr(network, 'adversatial_2')(batch_data, pred,reuse=True)
            al_false = tf.nn.softmax(al_false)

            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, model_path)
            print('Model restored from ', model_path)


            for Pid in range(len(subjects)):
                path = '/home/spc/Documents/AD/'+str(Pid)+'/'
                if not os.path.isdir(path):
                    os.mkdir(path)
                print(Pid)
                Patient_No = subjects[Pid]
                # Read input, label and location for each patient
                if datatype=='Sunny':
                    Input = h5f[dataset+'/input/%s/'%Patient_No][:,:,:,0]
                    Input = Input/np.max(Input)
                    Label = h5f[dataset+'/label/%s/'%Patient_No][:]
                    #Location = h5f[dataset+'/location/%s/'%Patient_No][:]
                else:
                    Input = h5f['/input/%s/'%Patient_No][:,:,:,0]
                    Input = Input/np.max(Input)
                    Label = h5f['/label/%s/'%Patient_No][:]
                    #Location = h5f['/location/%s/'%Patient_No][:]               
                #slice_idx_small = np.where(Location[:,1].reshape([-1])>10)[0]

                # Rescale the images
                #Input = rescale(Input, 0.8)
                #Label = rescale(Label, 0.8)
                # Resize the images if necessary

                # Make the prediction. From the predicted socre to generate final mask
                Input=Input[...,np.newaxis]

                num_batches = Input.shape[0]

                #var = [v for v in tf.global_variables() if v.name == 'prediction'][0]
                prediction_array = []
                true_list = np.zeros(num_batches)
                false_list = np.zeros(num_batches)
                for step in range(num_batches):
                    true_label, false_label, Labels, Preds, Images = sess.run(
                        [al_true, al_false, label, pred, batch_data], feed_dict={batch_data:Input[step:step+1], batch_label:Label[step:step+1]})
                    true_list[step] = np.argmax(true_label)
                    false_list[step] = 1-np.argmax(false_label)

                    name = '{0}_{1:.03}_{2:.03}_{3:.03}'.format(step,Jaccard(Labels[0,:,:,0],Preds[0,:,:,0]),true_label[0,0],false_label[0,0])
                    #name = '{0:.03}_{1:.03}_'.format(-np.mean(dice_list_epi), -np.mean(dice_list_endo))
                    #draw_output(p[:,:,1]+0.5*p[:,:,2], Input[i,:,:,0], Label[i,:,:,1]+0.5*Label[i,:,:,2], 
                    #    subject=Patient_No, name=i, dir_name=name)
                    """
                    plt.imshow(np.hstack((Images[0,:,:,0],Images[0,:,:,0],Images[0,:,:,0])), 'gray', interpolation='none')
                    label_all = np.hstack((np.zeros(Images[0,:,:,0].shape), Labels[0,:,:,0], Preds[0,:,:,0]))
                    mask = label_all.astype(np.int32)
                    masked = np.ma.masked_where(mask == 0, mask)
                    plt.imshow(masked, interpolation='none', alpha=0.4,cmap='hsv')

                    plt.axis('off')
                    plt.savefig(output_path + str(name) + '.png', transparent=True)
                    """

                    concat_img= np.hstack((Images[0,:,:,0], Labels[0,:,:,0], Preds[0,:,:,0]))*255
                    cv2.imwrite(path + str(name) + '.png', concat_img)
                    log_f.write('{0:.03}\t{1:.03}\t{2:.03}\n'.format(Jaccard(Labels[0,:,:,0],Preds[0,:,:,0]), true_label[0,0],false_label[0,0]))
                    #print(prediction_array.shape, prediction.shape)
                #print('{};  True_label: {};  False_Label: {}')
                print(np.mean(true_list), np.mean(false_list))

            log_f.close()
            if datatype!='Sunny':
                break

    #print('Result is:{}; {}; {}\n'.format(mean(dice_all_epi), mean(dice_all_endo), mean(dice_all_wall)))
    #print('Result is:{}; {}; {}\n'.format(mean(dice_all_patients_epi), mean(dice_all_patients_endo), mean(dice_all_patients_wall)))



if __name__=="__main__":
    #model_path = '/home/spc/Dropbox/GDCNN_test/output/sunny_out_test/\
#Sparse_fourier_new_7_294_sunny_channel32lr=1e-03drop=0.0_12-03_09.40.26/model/val_loss=.0.0251.hdf5'
    output_path = '/home/spc/Documents/AD/'
    data_path= '/media/DensoML/DENSO ML/LVData/LV_256_mix_eq_sel.h5'
    main(model_path, data_path, output_path, test_range=TEST_RANGE)
    #Predict_hdf5(model_path, data_path, output_path, test_range=TEST_RANGE)
