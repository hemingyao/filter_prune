import os
import tensorflow as tf
from six.moves import xrange
from flags import FLAGS, IMG_SIZE, DATA_AUG
from data_aug import *


class DataSet(object):
    def __init__(self, data_dir, data_range, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion
        self.data_range = data_range

    def get_filenames(self):
        """
        Customized for different dataset
        """
        #if self.subset in ['train', 'validation', 'test']:
        return [os.path.join(self.data_dir, '{}.tfrecord'.format(str(i)))
                for i in self.data_range]

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
            })
        #image = tf.decode_raw(features['image/encoded'], tf.uint8)
        image = tf.image.decode_png(features['image/encoded'], dtype=tf.uint8)

        #image.set_shape([DEPTH * HEIGHT * WIDTH])
        image.set_shape(IMG_SIZE)
        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(image, tf.float32)
        #image = tf.cast(
         #   tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
         #   tf.float32)
        #label = tf.cast(features['image/class/label'], tf.int32)
        label = features['image/class/label']
        label = tf.one_hot(label,10)
        # Custom preprocessing.
        image = self.preprocess(image)

        return image, label

    def make_batch(self, batch_size):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.contrib.data.TFRecordDataset(filenames).repeat()

        # Parse records.
        dataset = dataset.map(
            self.parser, num_threads=4, output_buffer_size=2 * batch_size)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = int(
                DataSet.num_examples_per_epoch(self.subset) * 0.1)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        # Batch it up.
        #dataset = dataset.prefetch(buffer_size=2 * batch_size)
        dataset = dataset.batch(batch_size)
        

        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            if 'random_crop' in DATA_AUG:
                image = random_crop(image, crop_padding_height=40, crop_padding_width=40)
            if 'random_flip_left_right' in DATA_AUG:
                image = random_flip_left_right(image)
            #image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            #image = tf.random_crop(image, IMG_SIZE)
            #image = tf.image.random_flip_left_right(image)
        return image

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return FLAGS.num_train_images
        elif subset == 'validation':
            return FLAGS.num_val_images
        elif subset == 'eval':
            return FLAGS.num_test_images
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


def input_fn(data_dir,
             subset,
             data_range,
             num_shards,
             batch_size,
             use_distortion_for_training=True):
    """Create input graph for model.
    Args:
      data_dir: Directory where TFRecords representing the dataset are located.
      subset: one of 'train', 'validate' and 'eval'.
      num_shards: num of towers participating in data-parallel training.
      batch_size: total batch size for training to be divided by the number of
      shards.
      use_distortion_for_training: True to use distortions.
    Returns:
      two lists of tensors for features and labels, each of num_shards length.
    """
    with tf.device('/cpu:0'):
        use_distortion = subset == 'train' and use_distortion_for_training
        dataset = DataSet(data_dir, data_range, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size)
        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [image_batch], [label_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        feature_shards = [[] for i in range(num_shards)]
        label_shards = [[] for i in range(num_shards)]
        for i in xrange(batch_size):
            idx = i % num_shards
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        return feature_shards, label_shards
