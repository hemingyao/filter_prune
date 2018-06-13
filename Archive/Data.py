from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import xrange
import os
import numpy as np
import tensorflow as tf
from flags import FLAGS
import random
import scipy.ndimage

_EPSILON = 1e-8
# --------------------------
#  Augmentation Computation
# --------------------------

class Augmentation:

	def __init__(self, batch):
		'''
		batch: 4-D tensor
		'''
		self.batch = batch

	def random_crop(self, crop_shape, padding=None):
		batch = self.batch
		oshape = np.shape(batch[0])
		if padding:
			oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
		new_batch = []
		npad = ((padding, padding), (padding, padding), (0, 0))
		for i in range(len(batch)):
			new_batch.append(batch[i])
			if padding:
				new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
				                          mode='constant', constant_values=0)
			nh = random.randint(0, oshape[0] - crop_shape[0])
			nw = random.randint(0, oshape[1] - crop_shape[1])
			new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
		                                nw:nw + crop_shape[1]]
		self.batch = new_batch

	def random_flip_leftright(self):
		batch = self.batch
		for i in range(len(batch)):
			if bool(random.getrandbits(1)):
				batch[i] = np.fliplr(batch[i])
		self.batch = batch

	def random_flip_updown(self):
		batch = self.batch
		for i in range(len(batch)):
			if bool(random.getrandbits(1)):
				batch[i] = np.flipud(batch[i])
		self.batch = batch

	def random_90degrees_rotation(self, rotations=[0, 1, 2, 3]):
		batch = self.batch
		for i in range(len(batch)):
			num_rotations = random.choice(rotations)
			batch[i] = np.rot90(batch[i], num_rotations)

		self.batch = batch

	def random_rotation(self, max_angle):
		batch = self.batch
		for i in range(len(batch)):
			if bool(random.getrandbits(1)):
				# Random angle
				angle = random.uniform(-max_angle, max_angle)
				batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle,
				                                              reshape=False)
				self.batch = batch

	def random_blur(self, sigma_max):
		batch = self.batch
		for i in range(len(batch)):
			if bool(random.getrandbits(1)):
				# Random sigma
				sigma = random.uniform(0., sigma_max)
				batch[i] = \
				    scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
		self.batch = batch


	def output(self):
		return self.batch


def _generate_image_and_label_batch(image, label, min_queue_examples,
									batch_size, shuffle):
	num_preprocess_threads = 5
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

	tf.summary.image('images', images)

	return images, tf.reshape(labels,[batch_size])


class ReadData:
	def __init__(self, status, shape):
		self.status = status
		self.image_shape = shape  # [height, width, depth]


	def read_from_files(self):
		if self.status=='train':
			filenames = [os.path.join(FLAGS.data_dir, 'data_batch_{}.bin'.format(i)) 
							for i in xrange(1,FLAGS.nbins)]

		elif self.status=='test':
			filenames = [os.path.join(FLAGS.data_dir, 'test_batch.bin')]
			
		else:
			filenames = [os.path.join(FLAGS.data_dir, 'validation_batch.bin')]

		for f in filenames:
			if not tf.gfile.Exists(f):
				raise ValueError('Failed to find file: ' + f)

		# Create a queue that produce the filenmaest to read
		filename_queue = tf.train.string_input_producer(filenames)
		label, image = self.read_data(filename_queue)

		# Image processing and augumentation
		image = tf.cast(image, tf.float32)
		image = tf.image.per_image_standardization(image)
		image = self.image_augumentation(image)

		# Set the shapes of tensors.
		image.set_shape(self.image_shape)
		label.set_shape([1])

		# Ensure that the random shuffling has good mixing properties.
		
		min_queue_examples = int(FLAGS.num_train_images * FLAGS.min_fraction_of_examples_in_queue)

		print('Filling {} queue. This will take a few minutes.'.format(self.status))

		# Generate a batch of images and labels by building up a queue of examples.
		if self.status=='train':
			return _generate_image_and_label_batch(image, label, min_queue_examples, 
											batch_size=FLAGS.train_batch_size, shuffle=True)
		else:
			return _generate_image_and_label_batch(image, label, min_queue_examples, 
											batch_size=FLAGS.train_batch_size, shuffle=False)




	def read_data(self, filename_queue):
		image_bytes = self.image_shape[0]*self.image_shape[1]*self.image_shape[2]
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
		# Convert from [depth, height, width] to [height, width, depth]
		uint8image = tf.transpose(image, [1, 2, 0])

		return label, uint8image



	def image_augumentation(self, image):
		
		image = tf.image.random_flip_left_right(image)

		# rotation
		#angle = random.randint(-10,10)
		#image = tf.contrib.image.rotate(image,angle)
		return image







