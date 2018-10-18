"""
Author: Heming Yao
System: Linux

Multiple data augmentation methods are implemented.

For the input 'img' in all function, it should be
A tensor of shape (num_images, num_rows, num_columns, num_channels) (NHWC), 
(num_rows, num_columns, num_channels) (HWC), or 
(num_rows, num_columns) (HW). 
The rank must be statically known (the shape is not TensorShape(None).

"""
import tensorflow as tf 
import random, math
import cv2
import numpy as np


def random_rotation(img, a=1):
    """ Random rotation
    a: range of the rotation angles
    """
    img = tf.contrib.image.rotate(img, 
            tf.random_uniform([1], minval=-a*math.pi, maxval=a*math.pi),
            interpolation='BILINEAR') 
    return img

def random_crop(img, crop_h, crop_w):
    """ Random crop
    crop_h: the height of the cropped image
    crop_w: the width of the cropped image
    """
    img_size = img.get_shape().as_list()
    img = tf.random_crop(img, [crop_h, crop_w, img_size[-1]])
    return img

def random_shift(img, max_h, max_w):
    """
    max_h: maximal shift distance in the height axis 
    max_w: maximal width distance in the height axis 
    """
    img_size = img.get_shape().as_list()
    height = img_size[0]
    width = img_size[1]
    img = tf.image.resize_image_with_crop_or_pad(img, 
                    target_height=height+2*max_h, target_width=width+2*max_w)
    img = tf.random_crop(img, img_size)
    return img


def random_flip_left_right(img):
    return tf.image.random_flip_left_right(img)
