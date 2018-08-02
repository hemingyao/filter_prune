from __future__ import division, print_function, absolute_import

import tflearn
import os
import numpy as np
import network 
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
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
img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = network.baseline_rescale(net, 0.5, 0.5, 0.0001, wd_scale=0, training_phase=True)
net = tf.nn.softmax(net)

mom = tflearn.Momentum(0.01, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar10',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=200, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
		  run_id='resnet_cifar10')

#pred_Y = model.predict(testX)
#pred_label = np.argmax(pred_Y,1)
#acc = np.mean(np.equal(pred_label, np.argmax(testY,1)))
#print(acc)
