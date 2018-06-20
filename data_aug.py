import tensorflow as tf 
import random, math
import cv2
import numpy as np

def random_rotation(img, a=1):
    img = tf.contrib.image.rotate(img, 
            tf.random_uniform([1], minval=-a*math.pi, maxval=a*math.pi)) 
    return img

def random_crop(img, crop_h, crop_w):
    img_size = img.get_shape().as_list()
    img = tf.random_crop(img, [crop_h, crop_w, img_size[-1]])
    return img

def random_shift(img, max_h, max_w):
    img_size = img.get_shape().as_list()
    height = img_size[0]
    width = img_size[1]
    img = tf.image.resize_image_with_crop_or_pad(img, 
                    target_height=height+2*max_h, target_width=width+2*max_w)
    img = tf.random_crop(img, img_size)
    return img



def random_flip_left_right(img):
    return tf.image.random_flip_left_right(img)
