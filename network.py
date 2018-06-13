import tensorflow as tf
import tflearn_dev as tflearn
import numpy as np


##################################################################################################################

def baseline_rescale(inputs, prob_fc, prob_conv, wd, wd_scale=0, training_phase=True):
    # First Block
    use_scale=False
    a = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    #a = [32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    conv = tflearn.conv_2d(inputs, a[0], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, 
                      scope="Conv"+'1_1', trainable=True)
    conv = tflearn.conv_2d(conv, a[1], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'1_2', trainable=True)
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_1')
    conv = tflearn.dropout(conv, prob_conv, name='dropout1')

    # Second Block
    
    conv = tflearn.conv_2d(conv, a[2], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'2_1', trainable=True)
    conv = tflearn.conv_2d(conv, a[3], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'2_2', trainable=True)
                      
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_2')
    conv = tflearn.dropout(conv, prob_conv, name='dropout2')

    # Third Block
    conv = tflearn.conv_2d(conv, a[4], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'3_1', trainable=True)
    
    conv = tflearn.conv_2d(conv, a[5], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'3_2', trainable=True)

    conv = tflearn.conv_2d(conv, a[6], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'3_3', trainable=True)

    
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_3')
    conv = tflearn.dropout(conv, prob_conv, name='dropout3')

    # Fourth Block
    
    conv = tflearn.conv_2d(conv, a[7], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'4_1', trainable=True)
    
    conv = tflearn.conv_2d(conv, a[8], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'4_2', trainable=True)
    conv = tflearn.conv_2d(conv, a[9], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'4_3', trainable=True)
                      
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_4')
    conv = tflearn.dropout(conv, prob_conv, name='dropout4')

    # Fifth Block
    
    conv = tflearn.conv_2d(conv, a[10], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'5_1', trainable=True)
    
    conv = tflearn.conv_2d(conv, a[11], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'5_2', trainable=True)
    conv = tflearn.conv_2d(conv, a[12], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'5_3', trainable=True)
                      
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_5')
    conv = tflearn.dropout(conv, prob_fc, name='dropout5')

    #conv = tf.reduce_mean(conv, [1,2], name='global_pool_1')
    conv = tflearn.fully_connected(conv, 512, activation='relu', scope='fc1', trainable=True)
    #conv = tf.nn.dropout(conv, prob_fc, name='dropout1')
    #conv = tflearn.fully_connected(conv, 4096, activation='relu', scope='fc2')
    conv = tflearn.dropout(conv, prob_fc, name='dropout6')
    conv = tflearn.fully_connected(conv, 10, name='fc3',activation='softmax', trainable=True)

    return conv


def baseline_rescale_save(inputs, prob_fc, prob_conv, wd, wd_scale, training_phase):
    # First Block
    use_scale=False
    a = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    #a = [32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]

    conv = tflearn.conv_2d_scale(inputs, a[0], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'1_1', am_scale=use_scale)
    conv = tflearn.conv_2d_scale(conv, a[1], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'1_2', am_scale=use_scale)
    conv = tf.contrib.layers.batch_norm(conv, is_training=training_phase, decay=0.99, center=True, scale=True,
                      scope='BN_1')
    conv = tflearn.max_pool_2d(conv, 2, name='pool_1')
    conv = tf.nn.dropout(conv, prob_conv, name='dropout1')

    # Second Block
    conv = tflearn.conv_2d_scale(conv, a[2], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'2_1', am_scale=use_scale)
    conv = tflearn.conv_2d_scale(conv, a[3], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'2_2', am_scale=use_scale)
    conv = tf.contrib.layers.batch_norm(conv, is_training=training_phase, decay=0.99, center=True, scale=True,
                      scope='BN_2')
    conv = tflearn.max_pool_2d(conv, 2, name='pool_2')
    conv = tf.nn.dropout(conv, prob_conv, name='dropout2')

    # Third Block
    conv = tflearn.conv_2d_scale(conv, a[4], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'3_1', am_scale=use_scale)
    conv = tflearn.conv_2d_scale(conv, a[5], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'3_2', am_scale=use_scale)
    conv = tflearn.conv_2d_scale(conv, a[6], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'3_3', am_scale=use_scale)
    conv = tf.contrib.layers.batch_norm(conv, is_training=training_phase, decay=0.99, center=True, scale=True,
                      scope='BN_3')
    conv = tflearn.max_pool_2d(conv, 2, name='pool_3')
    conv = tf.nn.dropout(conv, prob_conv, name='dropout3')

    # Fourth Block
    conv = tflearn.conv_2d_scale(conv, a[7], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'4_1', am_scale=use_scale)
    conv = tflearn.conv_2d_scale(conv, a[8], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'4_2', am_scale=use_scale)
    conv = tflearn.conv_2d_scale(conv, a[9], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'4_3', am_scale=use_scale)
    conv = tf.contrib.layers.batch_norm(conv, is_training=training_phase, decay=0.99, center=True, scale=True,
                      scope='BN_4')
    conv = tflearn.max_pool_2d(conv, 2, name='pool_4')
    conv = tf.nn.dropout(conv, prob_conv, name='dropout4')

    # Fifth Block
    conv = tflearn.conv_2d_scale(conv, a[10], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'5_1', am_scale=use_scale)
    conv = tflearn.conv_2d_scale(conv, a[11], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'5_2', am_scale=use_scale)
    conv = tflearn.conv_2d_scale(conv, a[12], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, wd_scale=wd_scale,
                      scope="Conv"+'5_3', am_scale=use_scale)
    conv = tf.contrib.layers.batch_norm(conv, is_training=training_phase, decay=0.99, center=True, scale=True,
                      scope='BN_5')
    conv = tflearn.max_pool_2d(conv, 2, name='pool_5')
    conv = tf.nn.dropout(conv, prob_fc, name='dropout5')

    #conv = tf.reduce_mean(conv, [1,2], name='global_pool_1')
    conv = tflearn.fully_connected(conv, 512, activation='relu', scope='fc1')
    #conv = tf.nn.dropout(conv, prob_fc, name='dropout1')
    #conv = tflearn.fully_connected(conv, 4096, activation='relu', scope='fc2')
    conv = tf.nn.dropout(conv, prob_fc, name='dropout6')
    conv = tflearn.fully_connected(conv, 10, name='fc3')
    #x = tf.nn.softmax(x, name='softmax')
    return conv
    