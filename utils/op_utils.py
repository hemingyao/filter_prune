"""
Author: Heming Yao
System: Linux

Utils.
"""

import tensorflow as tf
import numpy as np
import sys, cv2, os, pickle
import scipy.io
import glob

_EPSILON = 1e-8


def weighted_cross_entropy(y_preds, y_trues):
    """ loss function for image segmentation
    Args:
      y_preds: a tensor of shape [batch, height, width, channel]
      y_preds: a tensor of shape [batch, height, width, channel]
    """
    loss = 0
    weights = [1, 5, 5]
    for i in range(3):
        y_pred = y_preds[:,:,:,i]
        y_true = y_trues[:,:,:,i]

        y_pred = tf.contrib.layers.flatten(y_pred)
        y_true = tf.contrib.layers.flatten(y_true)
        loss = loss + tf.reduce_mean(-1*y_true*tf.map_fn(tf.log, y_pred+1e-3))*weights[i]
    return loss


def weighted_jaccard_loss(y_preds, y_trues):
    """ loss function for image segmentation
    Args:
      y_preds: a tensor of shape [batch, height, width, channel]
      y_preds: a tensor of shape [batch, height, width, channel]
    """
    dice = 0
    weights = [1, 1]
    for i in range(2):
        y_pred = y_preds[:,:,:,i+1]
        y_true = y_trues[:,:,:,i+1]
        union = y_pred+y_true - y_pred*y_true
        dice_list = -1*tf.reduce_sum(y_pred*y_true,(1,2))/(tf.reduce_sum(union,(1,2))+1)
        dice = dice+tf.reduce_mean(dice_list)
    return dice


def weighted_dice_loss(y_preds, y_trues):
    """ loss function for image segmentation
    Args:
      y_preds: a tensor of shape [batch, height, width, channel]
      y_preds: a tensor of shape [batch, height, width, channel]
    """
    dice = 0
    weights = [1, 1]
    for i in range(2):
        y_pred = y_preds[:,:,:,i+1]
        y_true = y_trues[:,:,:,i+1]
        union = y_pred+y_true
        dice_list = -2*(tf.reduce_sum(y_pred*y_true,(1,2))+0.5)/(tf.reduce_sum(union,(1,2))+1)
        #dice_list = -2*(tf.reduce_sum(y_pred*y_true,(1,2))+0.5)/(tf.reduce_sum(union,(1,2))+1)
        dice = dice+tf.reduce_mean(dice_list)*weights[i]
    return dice


def dice(y_pred, y_true, whole=False):
    """ Dice calculation for image segmentation
    Args:
      y_preds: a numpy array of shape [batch, height, width, channel]
      y_preds: a numpy array of shape [batch, height, width, channel]
    """
    y_true = y_true.astype(np.float)
    union = y_pred+y_true
    if whole:
        dice = 2*np.sum(y_pred*y_true, (0, 1, 2))/(np.sum(union, (0, 1,2))+0.00001)
    else:
        dice = 2*np.sum(y_pred*y_true, (1, 2))/(np.sum(union, (1, 2))+0.00001)
    return np.mean(dice)


def jaccard(y_pred, y_true, all=False):
    """ Jaccard calculation for image segmentation
    Args:
      y_preds: a numpy array of shape [batch, height, width, channel]
      y_preds: a numpy array of shape [batch, height, width, channel]
    """
    y_true = y_true.astype(np.float)
    union = y_pred+y_true - y_pred*y_true
    if all:
        jaccard = np.sum(y_pred*y_true, (0,1,2))/(np.sum(union, (0,1,2))+0.00001)
    else:
        jaccard = np.sum(y_pred*y_true, (1,2))/(np.sum(union, (1,2))+0.00001)
    return np.mean(jaccard)


def samplewise_zero_center(batch, per_channel=False):
    """ Center the data
    batch: a numpy array
    """
    for i in range(len(batch)):
        if not per_channel:
            batch[i] -= np.mean(batch[i])
        else:
            batch[i] -= np.mean(batch[i], axis=(0, 1, 2), keepdims=True)
    return batch


def samplewise_stdnorm(batch):
    """ Standarlize the data
    batch: a numpy array
    """
    for i in range(len(batch)):
        batch[i] /= (np.std(batch[i], axis=0) + _EPSILON)
    return batch


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 



def shuffle(*arrs):
    """ shuffle.
    Shuffle given arrays at unison, along first axis.
    Arguments:
        *arrs: Each array to shuffle at unison.
    Returns:
        Tuple of shuffled arrays.
    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


def to_categorical(y, nb_classes):
    """ Convert labels to one-hot coding. 
    """
    y = np.asarray(y, dtype='int32')
    # high dimensional array warning
    if len(y.shape) > 2:
        warnings.warn('{}-dimensional array is used as input array.'.format(len(y.shape)), stacklevel=2)
    # flatten high dimensional array
    if len(y.shape) > 1:
        y = y.reshape(-1)
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)),y] = 1.
    return Y


def calculate_number_of_parameters(variables, show=False):
    """ Calculate the total number of variables in the network
    """
    total_parameters = 0
    for variable in variables:
        #print(variable)
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        #print(variable_parameters)
    return total_parameters


def add_scaled_noise_to_gradients(self, grads_and_vars, gradient_noise_scale):
    """Adds scaled noise from a 0-mean normal distribution to gradients."""
    gradients, variables = zip(*grads_and_vars)
    noisy_gradients = []
    for gradient in gradients:
        if gradient is None:
            noisy_gradients.append(None)
            continue
        if isinstance(gradient, ops.IndexedSlices):
            gradient_shape = gradient.dense_shape
        else:
            gradient_shape = gradient.get_shape()
        noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
        noisy_gradients.append(gradient + noise)
    return list(zip(noisy_gradients, variables))

