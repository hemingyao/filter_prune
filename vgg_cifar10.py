from __future__ import division, print_function, absolute_import

import tflearn
import os
import numpy as np
from network import baseline_rescale
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="3"
# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
#tflearn.config.init_graph (seed=1, log_device=False, num_cores=4, gpu_memory_fraction=0.8, soft_placement=True)
n = 5

# Data loading
from tflearn.datasets import cifar10

(X, Y), (testX, testY) = cifar10.load_data()
Y = tflearn.data_utils.to_categorical(Y, 10)
testY = tflearn.data_utils.to_categorical(testY, 10)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
#img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
conv = baseline_rescale(net, prob_fc=0.5,  prob_conv=0.5, wd=0.0005)

mom = tflearn.Momentum(0.01, lr_decay=0.1, decay_step=32000, staircase=True)
#mom = tflearn.SGD(0.01)
net = tflearn.regression(conv, optimizer=mom,
                         loss='categorical_crossentropy')
# Training

model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar10',
                    max_checkpoints=2, tensorboard_verbose=2, tensorboard_dir='./output/',
                    clip_gradients=0)

model.fit(X, Y, n_epoch=50, validation_set=(testX[1:100,:], testY[1:100,:]),
          snapshot_epoch=False, snapshot_step=400,
          show_metric=True, batch_size=128, shuffle=False,
          run_id='resnet_cifar10')


#pred_Y = model.predict(testX)
#pred_label = np.argmax(pred_Y,1)
#acc = np.mean(np.equal(pred_label, np.argmax(testY,1)))
#print(acc)
