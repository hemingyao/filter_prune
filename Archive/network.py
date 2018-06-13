import tensorflow as tf
import tflearn as tflearn
import numpy as np


##################################################################################################################

def bn_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, np.arange(len(shape)-1), keep_dims=True)
            avg=tf.reshape(avg, [avg.shape.as_list()[-1]])
            var=tf.reshape(var, [var.shape.as_list()[-1]])
            #update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_avg=tf.assign(moving_avg, moving_avg*decay+avg*(1-decay))
            #update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            update_moving_var=tf.assign(moving_var, moving_var*decay+var*(1-decay))
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output


def bn_layer_top(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the 
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    #assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(
        is_training,
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )



def baseline_rescale(inputs, prob_fc, prob_conv, wd, wd_scale, training_phase):
    # First Block
    use_scale=False
    a = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    #a = [32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    conv = tflearn.conv_2d(inputs, a[0], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd, 
                      scope="Conv"+'1_1')
    conv = tflearn.conv_2d(conv, a[1], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'1_2')
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_1')
    conv = tflearn.dropout(conv, prob_conv, name='dropout1')

    # Second Block
    conv = tflearn.conv_2d(conv, a[2], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'2_1')
    conv = tflearn.conv_2d(conv, a[3], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'2_2')
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_2')
    conv = tflearn.dropout(conv, prob_conv, name='dropout2')

    # Third Block
    conv = tflearn.conv_2d(conv, a[4], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'3_1')
    conv = tflearn.conv_2d(conv, a[5], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'3_2')
    conv = tflearn.conv_2d(conv, a[6], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'3_3')
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_3')
    conv = tflearn.dropout(conv, prob_conv, name='dropout3')

    # Fourth Block
    conv = tflearn.conv_2d(conv, a[7], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'4_1')
    conv = tflearn.conv_2d(conv, a[8], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'4_2')
    conv = tflearn.conv_2d(conv, a[9], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'4_3')
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_4')
    conv = tflearn.dropout(conv, prob_conv, name='dropout4')

    # Fifth Block
    conv = tflearn.conv_2d(conv, a[10], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'5_1')
    conv = tflearn.conv_2d(conv, a[11], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'5_2')
    conv = tflearn.conv_2d(conv, a[12], [3,3], activation='relu', weights_init='normal',regularizer='L2',
                      weight_decay=wd,
                      scope="Conv"+'5_3')
    conv = tflearn.batch_normalization(conv)
    conv = tflearn.max_pool_2d(conv, 2, name='pool_5')
    conv = tflearn.dropout(conv, prob_fc, name='dropout5')

    #conv = tf.reduce_mean(conv, [1,2], name='global_pool_1')
    conv = tflearn.fully_connected(conv, 512, activation='relu', scope='fc1')
    #conv = tf.nn.dropout(conv, prob_fc, name='dropout1')
    #conv = tflearn.fully_connected(conv, 4096, activation='relu', scope='fc2')
    conv = tflearn.dropout(conv, prob_fc, name='dropout6')
    conv = tflearn.fully_connected(conv, 10, name='fc3')

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
    