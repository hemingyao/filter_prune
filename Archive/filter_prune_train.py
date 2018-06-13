#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from datetime import datetime
import re
import time
import os
import time
from tflearn.datasets import cifar10

## Import Customized Functions
#import Model
from Data import ReadData, Augmentation
from flags import FLAGS
import tflearn
from tensorflow.python.framework import ops
from utils import *
import network

_FLOATX = tf.float32
_EPSILON = 1e-10

class Train():
	def __init__(self, model, img_size, learning_rate, run_id, config, weight_decay):
		self.model = model
		self.img_size = img_size
		self.learning_rate = learning_rate
		self.run_id = run_id
		self.dict_widx = None
		self.tf_config = config
		self.wd = weight_decay
		

	def _placeholders(self):
		"""
		Define Placeholders used to building the graph
		"""
		self.batch_data = tf.placeholder(name='data_pl', dtype=tf.float32,
			shape=(FLAGS.train_batch_size,)+self.img_size)
		self.batch_labels = tf.placeholder(name='label_pl', dtype=tf.float32,
			shape=(FLAGS.train_batch_size,10))


	def _build_graph(self):
		global_step = tf.contrib.framework.get_or_create_global_step()

		# Calculate logits using training data and vali data seperately
		logits = getattr(network, 'baseline_rescale')(self.batch_data, 0.5, 0.5, 0.0005, 0, True)

		with tf.name_scope("Crossentropy"):
			#logits /= tf.reduce_sum(logits,reduction_indices=len(logits.get_shape())-1, keep_dims=True)
			# manual computation of crossentropy
			#logits = tf.clip_by_value(logits, tf.cast(_EPSILON, dtype=_FLOATX),
		    #                      tf.cast(1.-_EPSILON, dtype=_FLOATX))
			#cross_entropy = - tf.reduce_mean(self.batch_labels * tf.log(logits))
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				labels=self.batch_labels, logits=logits, name='corss_entropy'))

			weight_loss = tf.get_collection('losses')
			self.loss = cross_entropy
			if len(weight_loss)>0:
				self.total_loss = tf.add(tf.divide(tf.add_n(weight_loss), FLAGS.train_batch_size), cross_entropy, name='total_loss')
			else:
				self.total_loss = cross_entropy


		with tf.name_scope("train"):
			opt = tf.train.AdamOptimizer(self.learning_rate)
			grads_and_vars = opt.compute_gradients(self.total_loss)

			if self.dict_widx is None:
				apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
			else:
				grads_and_vars = apply_prune_on_grads(grads_and_vars, self.dict_widx)
				apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

			for var in tf.trainable_variables():
				tf.summary.histogram(var.op.name, var)

		with tf.name_scope("accuracy"):
			logits = tf.nn.softmax(logits)
			self.prediction = tf.argmax(logits,1)
			correct_prediction = tf.equal(self.prediction, tf.argmax(self.batch_labels,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.train_op = tf.group(apply_gradient_op)
		self.summary_op = tf.summary.merge_all()

		
	def train(self, **kwargs):
		ops.reset_default_graph()
		sess = tf.Session(config=self.tf_config)
		with sess.as_default():
			(X, Y), (X_test, Y_test) = cifar10.load_data()
			X, Y = shuffle(X, Y)
			X = samplewise_zero_center(X)
			X = samplewise_stdnorm(X)
			X_test = samplewise_zero_center(X_test)
			X_test = samplewise_stdnorm(X_test)
			Y = to_categorical(Y, 10)
			Y_test = to_categorical(Y_test, 10)

			self.train_data, self.train_label = np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)
			self.vali_data, self.vali_label = np.array(X_test, dtype=np.float32), np.array(Y_test, dtype=np.int32)

			print(len(self.train_data))
			# Data Reading objects
			self._placeholders()
			if len(kwargs)==0:
				self.dict_widx = None
				self._build_graph()
				self.saver = tf.train.Saver(tf.global_variables())
				# Build an initialization operation to run below
				init = tf.global_variables_initializer()
				sess.run(init)

			else:
				self.dict_widx = kwargs['dict_widx']
				pruned_model = kwargs['pruned_model_path']
				tflearn.config.init_training_mode()
				self._build_graph()				
				init = tf.global_variables_initializer()
				sess.run(init)

				self.saver = tf.train.Saver(tf.global_variables())
				self.saver.restore(sess, pruned_model)
				print('Pruned model restored from ', pruned_model)


			# This summary writer object helps write summaries on tensorboard
			summary_writer = tf.summary.FileWriter(FLAGS.log_dir+self.run_id)
			summary_writer.add_graph(sess.graph)

			train_error_list = []
			val_error_list = []

			print('Start training...')
			print('----------------------------------')


			train_steps_per_epoch = FLAGS.num_train_images//FLAGS.train_batch_size
			report_freq = train_steps_per_epoch

			train_steps = FLAGS.train_epoch * train_steps_per_epoch

			durations = []
			train_loss_list = []
			train_accuracy_list = []
			train_total_loss_list = []

			best_accuracy = 0

			log_file = open(os.path.join(FLAGS.log_dir, self.run_id, 'loss_accuracy_list'), 'a')

			for step in range(train_steps):
				tflearn.is_training(True)

				start_time = time.time()

				train_batch_data, train_batch_labels = self._generate_batch(self.train_data, self.train_label,
													   FLAGS.train_batch_size, step, train=True)
				
				_, summary_str, loss_value, total_loss, accuracy = sess.run([self.train_op, self.summary_op, self.loss, self.total_loss, self.accuracy], 
													   feed_dict={self.batch_data: train_batch_data,
													   			  self.batch_labels: train_batch_labels})

				#summary_str = ''
				duration = time.time() - start_time
				durations.append(duration)
				#print(duration)
				train_loss_list.append(loss_value)
				train_total_loss_list.append(total_loss)
				train_accuracy_list.append(accuracy)

				assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
				
				
				if step%report_freq == 0:
					summary_writer.add_summary(summary_str, step)

					sec_per_report = np.sum(np.array(durations))*1.2
					train_loss_value = np.mean(np.array(train_loss_list))
					train_total_loss = np.mean(np.array(train_total_loss_list))
					train_accuracy_value = np.mean(np.array(train_accuracy_list))

					train_loss_list = []
					train_total_loss_list = []
					train_accuracy_list = []
					durations = []

					train_summ = tf.Summary()
					train_summ.value.add(tag="train_loss", simple_value=train_loss_value.astype(np.float))
					train_summ.value.add(tag="train_total_loss", simple_value=train_total_loss.astype(np.float))
					train_summ.value.add(tag="train_accuracy", simple_value=train_accuracy_value.astype(np.float))

					summary_writer.add_summary(train_summ, step)
                                                      
					vali_loss_value, vali_accuracy_value = self._full_validation(sess)

					if vali_accuracy_value>best_accuracy:
						best_accuracy = vali_accuracy_value

						model_dir = os.path.join(FLAGS.log_dir, self.run_id, 'model')
						if not os.path.isdir(model_dir):
							os.mkdir(model_dir)
						checkpoint_path = os.path.join(model_dir, 'vali_{:.3f}'.format(vali_accuracy_value))

						self.saver.save(sess, checkpoint_path, global_step=step)


					vali_summ = tf.Summary()
					vali_summ.value.add(tag="vali_loss", simple_value=vali_loss_value.astype(np.float))
					vali_summ.value.add(tag="vali_accuracy", simple_value=vali_accuracy_value.astype(np.float))

					summary_writer.add_summary(vali_summ, step)
					summary_writer.flush()

					format_str = ('Epoch %d, loss = %.4f, total_loss = %.4f, accuracy = %.4f, vali_loss = %.4f, vali_accuracy = %.4f (%.3f ' 'sec/report)')
					print(format_str % (step//report_freq, train_loss_value, train_total_loss, train_accuracy_value, vali_loss_value, vali_accuracy_value, sec_per_report))
					log_file.write('{},{},{},{} \n'.format(train_loss_value, train_accuracy_value, vali_loss_value, vali_accuracy_value))
			log_file.close()
				

	def test(self):
		ops.reset_default_graph()
		(X, Y), (X_test, Y_test) = cifar10.load_data()
		X, Y = shuffle(X, Y)
		X = samplewise_zero_center(X)
		X = samplewise_stdnorm(X)
		X_test = samplewise_zero_center(X_test)
		X_test = samplewise_stdnorm(X_test)

		test_batch_data = tf.placeholder(name='data_pl', dtype=tf.float32,
			shape=(FLAGS.test_batch_size,)+self.img_size)
		test_batch_labels = tf.placeholder(name='label_pl', dtype=tf.int32,
			shape=(FLAGS.test_batch_size,))

		logits = getattr(Model, self.model)(test_batch_data)
		prediction = tf.cast(tf.argmax(logits,1), tf.int32)
		correct_prediction = tf.equal(prediction, test_batch_labels)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		saver = tf.train.Saver(tf.global_variables())
		sess = tf.Session(config=self.tf_config)

		saver.restore(sess, FLAGS.test_ckpt_path)
		print('Model restored from ', FLAGS.test_ckpt_path)

		prediction_array = np.array([]).reshape(-1, FLAGS.num_categories)

		num_batches = FLAGS.num_test_images//FLAGS.test_batch_size
		accuracy_list = []

		for step in range(num_batches):
			batch_data, batch_labels = self._generate_batch(X_test, Y_test,
													   FLAGS.test_batch_size, step, train=False)
				

			batch_prediction_array, batch_accuracy = sess.run([prediction, accuracy], feed_dict={test_batch_data: batch_data,
													   			  test_batch_labels: test_batch_labels})
			#prediction_array = np.concatenate((prediction_array, batch_prediction_array))
			accuracy_list.append(batch_accuracy)

		accuracy = np.mean(np.array(accuracy_list, dtype=np.float32))

		return prediction_array, accuracy_list


	def _full_validation(self, sess):
		tflearn.is_training(True)
		num_batches_vali = FLAGS.num_eval_images // FLAGS.test_batch_size

		loss_list = []
		accuracy_list = []

		start_time = time.time()

		for step_vali in range(num_batches_vali):
			vali_batch_data, vali_batch_labels = self._generate_batch(self.vali_data, self.vali_label, 
													  			 	  FLAGS.train_batch_size, step_vali, train=False)	

			loss, accuracy = sess.run([self.loss, self.accuracy],  feed_dict={self.batch_data: vali_batch_data,
													   	                  self.batch_labels: vali_batch_labels})

			loss_list.append(loss)
			accuracy_list.append(accuracy)

		duration = time.time() - start_time

		vali_loss_value = np.mean(np.array(loss_list))
		vali_accuracy_value = np.mean(np.array(accuracy_list))

		return vali_loss_value, vali_accuracy_value


	def _generate_batch(self, all_data, all_labels, batch_size, step, train=True):
		train_steps_per_epoch = FLAGS.num_train_images//FLAGS.train_batch_size
		if step % train_steps_per_epoch ==0:
			all_data, all_labels = shuffle(all_data, all_labels)
		num_batches = FLAGS.num_train_images//batch_size
		offset = step%num_batches

		batch_data = np.copy(all_data[offset*batch_size:(offset+1)*batch_size, :])
		#batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

		#batch_data = whitening_image(batch_data)
		batch_labels = np.copy(all_labels[offset*batch_size:(offset+1)*batch_size,:])

		#batch_data = samplewise_zero_center(batch_data)
		#batch_data = samplewise_stdnorm(batch_data)
		'''
		if train:
			aug = Augmentation(batch_data)

			aug.random_flip_leftright()
			aug.random_rotation(max_angle=25.)
			batch_data = aug.output()
		'''
		return batch_data, batch_labels



def main(argv=None):
	os.environ["CUDA_VISIBLE_DEVICES"]="3" 
	tf_config=tf.ConfigProto() 
	tf_config.gpu_options.allow_growth=True 
	tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
	img_size = (32,32,3)

	model = 'vgg16_all'
	learning_rate = 0.01
	option = 1
	weight_decay = 0

	if option == 1:
		run_name = 'test'
		run_id = '{}_{}_lr_{}_wd_{}_{}'.format(model, run_name, learning_rate, weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))

		# First Training
		train = Train(model, img_size, learning_rate, run_id, tf_config, weight_decay)
		train.train()

	elif option == 2:
		# Filter pruning
		trained_path = '/home/spc/Dropbox/tf_GDCNN/log/model/Test_0.0001_Aug_07_18_28_33_vali_0.8699822425842285-22320'
		layers = getattr(Model, model+'_variable_of_interest')()
		sel_layer_names = layers[2:len(layers)]
		run_name = 'Prune_Test'
		run_id = '{}_{}'.format(run_name,time.strftime("%b_%d_%H_%M_%S", time.gmtime()))

		dict_widx, pruned_model_path = apply_pruning(sel_layer_names, trained_path, run_id, tf_config)
		
		train = Train(model, img_size, learning_rate, run_id='{}_retrain_{}'.format(run_id, learning_rate), config=tf_config)
		train.train(dict_widx=dict_widx, pruned_model_path=pruned_model_path)

	elif option == 3:
		train = Train(model, img_size, learning_rate, run_name, tf_config)
		_, accuracy = train.test()
		print(accuracy)

if __name__ == '__main__':
	tf.app.run()
