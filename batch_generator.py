from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy
import numpy as np
import h5py
from flags import FLAGS
import multiprocessing as mp
from scipy.misc import imresize
import os
import tensorflow as tf
import random

def xslice(arr, slices):
    if isinstance(slices, tuple):
        return sum((arr[s] if isinstance(s, slice) else [arr[s]] for s in slices), [])
    elif isinstance(slices, slice):
        return arr[slices]
    else:
        return [arr[slices]]

def to_categorical(y, nb_classes):
    """ to_categorical.

    Convert class vector (integers from 0 to 
    nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.

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

    
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle, num_preprocess_threads):
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image,label], 
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + num_preprocess_threads * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.batch(
            [image,label], 
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + num_preprocess_threads * batch_size)

    #tf.summary.image('images', images)

    return images, tf.reshape(labels,[batch_size,-1])


class ReadData:
    def __init__(self, status, shape, set_id, subject_range):
        self.status = status
        self.image_shape = shape 
        self.set_id = set_id
        self.range = subject_range


    def read_from_files(self):
        data_dir = os.path.join(FLAGS.data_dir, self.set_id)
        filenames = os.listdir(data_dir)
        subject_index = self.range
        """
        if self.status=='test':
            subject_index = xslice(TRAIN,self.range)
        else:
            subject_index = xslice(TRAIN,self.range)
        """
        print(subject_index)
        #subject_index = TRAIN

        filenames = [os.path.join(data_dir, 'data_batch_{}.bin'.format(i))
                                    for i in subject_index]

        # Create a queue that produce the filenmaest to read
        filename_queue = tf.train.string_input_producer(filenames)
        label, image = self.read_data(filename_queue)
        image = tf.cast(image, tf.float32)/255

        label = tf.cast(label, tf.int32)
        ## Image processing and augumentation
        # Image is a 4D tensor: [depth, height, width, channel]
        #if len(self.image_shape)>3:
            # We want to remove the channel dimension as the image is grayscale
        #    image = tf.squeeze(image, [-1])

            # Convert from [depth, height, width] to [height, width, depth]
        #    image = tf.transpose(image, [1, 2, 0])

        #image = tf.image.per_image_standardization(image) 
        #image = image - 0.49

        #if self.status=='train':
        #   image = self.image_augumentation(image)   #TODO
        #image = (image - tf.reduce_mean(image))
        #mean = 120.707
        #std = 64.15
        #image = (image-mean)/(std+1e-7)


        #if len(self.image_shape)>3:
            # Convert from [height, width, depth] to [depth, height, width]
        #image = tf.transpose(image, [2, 0, 1])
        #image = tf.expand_dims(image,-1)
        
        # Set the shapes of tensors.
        image.set_shape(self.image_shape)
        label.set_shape([FLAGS.label_bytes])
        label = tf.one_hot(label, 10)

        #if self.status=='train':
        #    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1, dtype=tf.float32)
        #    image = image + noise

        # Ensure that the random shuffling has good mixing properties.
        
        min_queue_examples = int(FLAGS.num_train_images * FLAGS.min_fraction_of_examples_in_queue)

        print('Filling {} queue. This will take a few minutes.'.format(self.status))

        # Generate a batch of images and labels by building up a queue of examples.
        if self.status=='train':
            return _generate_image_and_label_batch(image, label, min_queue_examples, 
                                            batch_size=FLAGS.batch_size, shuffle=False, num_preprocess_threads=4)
        elif self.status=='validation':
            return _generate_image_and_label_batch(image, label, min_queue_examples, 
                                            batch_size=FLAGS.val_batch_size, shuffle=False, num_preprocess_threads=4)
        elif self.status=='test':
            return _generate_image_and_label_batch(image, label, min_queue_examples, 
                                            batch_size=FLAGS.val_batch_size, shuffle=False, num_preprocess_threads=1)




    def read_data(self, filename_queue):
        image_bytes = 1
        for elem in self.image_shape:
            image_bytes *= elem

        label_bytes = FLAGS.label_bytes

        # Every record consists of a label followed by the image, with a
            # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represents the label
        label = tf.cast(
            tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        image = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes], [label_bytes+image_bytes]), 
            [self.image_shape[2], self.image_shape[0], self.image_shape[1]])
        # Convert from [depth, height, width, channel] to [height, width, depth]
        
        image = tf.transpose(image, [1, 2, 0])
        return label, image


    def image_augumentation(self, image):
        
        #image = tf.image.random_flip_left_right(image)
        image = tf.random_crop(image, [28, 28, 3])
        paddings = tf.constant([[2, 2], [2, 2]])
        image = tf.pad(image, paddings, "CONSTANT")

        #image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        #angle = random.randint(-30,30)/180*3.14
        #image = tf.contrib.image.rotate(image,angle)
        # rotation
        return image