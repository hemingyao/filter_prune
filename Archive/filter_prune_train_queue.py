#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from datetime import datetime
import re
import time
import os
import time

## Import Customized Functions
import Model
from Data import ReadData
from flags import FLAGS
import tflearn_dev
from utils import apply_pruning_scale, apply_prune_on_grads

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


	def _build_graph(self):
		global_step = tf.contrib.framework.get_or_create_global_step()

		# Calculate logits using training data and vali data seperately
		logits = getattr(Model, self.model)(self.batch_data, self.wd)
		
		with tf.name_scope("loss"):

			logits /= tf.reduce_sum(logits,reduction_indices=len(logits.get_shape())-1, keep_dims=True)
			# manual computation of crossentropy
			logits = tf.clip_by_value(logits, tf.cast(_EPSILON, dtype=_FLOATX),
		                          tf.cast(1.-_EPSILON, dtype=_FLOATX))
			loss = - tf.reduce_sum(tf.cast(self.batch_labels, tf.float32) * tf.log(logits))

			#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			#	labels=self.batch_labels, logits=logits, name='corss_entropy'))

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
			self.prediction = tf.cast(tf.argmax(tf.nn.softmax(logits),1), tf.int32)
			correct_prediction = tf.equal(self.prediction, self.batch_labels)
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.train_op = tf.group(apply_gradient_op)
		self.summary_op = tf.summary.merge_all()

		
	def train(self, **kwargs):
		#with tf.Graph().as_default():
		ops.reset_default_graph()
		sess = tf.Session(config=self.tf_config)

		with sess.as_default():
			# Data Reading objects
			train_data = ReadData(status='train', shape=self.img_size)
			vali_data = ReadData(status='validation', shape=self.img_size)

			train_batch_data, train_batch_labels = train_data.read_from_files()
			vali_batch_data, vali_batch_labels = vali_data.read_from_files()
			self.am_training = tf.placeholder(dtype=bool, shape=())
			self.batch_data = tf.cond(self.am_training, lambda:train_batch_data, lambda:vali_batch_data)
			self.batch_labels = tf.cond(self.am_training, lambda:train_batch_labels, lambda:vali_batch_labels)

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
				
				self._build_graph()			
				tflearn_dev.config.init_training_mode()	
				init = tf.global_variables_initializer()
				sess.run(init)

				self.saver = tf.train.Saver(tf.global_variables())
				self.saver.restore(sess, pruned_model)
				print('Pruned model restored from ', pruned_model)


			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

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

			tflearn_dev.is_training(True)

			log_file = open(os.path.join(FLAGS.log_dir, self.run_id, 'loss_accuracy_list'), 'a')
			for step in range(train_steps):
				#print('{} step starts'.format(step))
				best_accuracy = 0

				start_time = time.time()

				_, summary_str, loss_value, total_loss, accuracy = sess.run([self.train_op, self.summary_op, self.loss, self.total_loss, self.accuracy], 
														feed_dict={self.am_training: True})
				#sess.run(self.batch_labels,feed_dict={self.am_training: True})

				duration = time.time() - start_time
				durations.append(duration)
				#print(duration)
				train_loss_list.append(loss_value)
				train_total_loss_list.append(total_loss)
				train_accuracy_list.append(accuracy)
				
				
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
                                                      
					vali_loss_value, vali_accuracy_value = self._full_validation(vali_data, sess)

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
		test_data = ReadData(status='test', shape=self.img_size)
		test_batch_data, test_batch_labels = test_data.read_from_files()

		logits = getattr(Model, self.model)(test_batch_data)
		prediction = tf.nn.softmax(logits,1)
		correct_prediction = tf.equal(tf.argmax(logits,1), test_batch_labels)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		saver = tf.train.Saver(tf.all_variables())
		sess = tf.Session(config=self.tf_config)

		saver.restore(sess, FLAGS.test_ckpt_path)
		print('Model restored from ', FLAGS.test_ckpt_path)

		prediction_array = np.array([]).reshape(-1, FLAGS.num_categories)

		num_batches = FLAGS.num_test_images//FLAGS.test_batch_size
		accuracy_list = []

		for step in range(num_batches):
			
			batch_prediction_array, batch_accuracy = sess.run(
				[self.prediction, self.accuracy])
			prediction_array = np.concatenate((prediction_array, batch_prediction_array))
			accuracy_list.append(accuracy)

		accuracy = np.mean(np.array(accuracy_list, dtype=np.float32))

		return prediction_array, accuracy_list


	def _full_validation(self, vali_data, sess):
		tflearn_dev.is_training(True)
		num_batches_vali = FLAGS.num_eval_images // FLAGS.test_batch_size

		loss_list = []
		accuracy_list = []

		start_time = time.time()

		for step_vali in range(num_batches_vali):
			loss, accuracy = sess.run([self.loss, self.accuracy], 
											feed_dict={self.am_training: False})
			
			#sess.run([self.batch_data, self.batch_labels], 
			#								feed_dict={self.am_training: False})


			loss_list.append(loss)
			accuracy_list.append(accuracy)
		duration = time.time() - start_time
		#print(duration)

		vali_loss_value = np.mean(np.array(loss_list))
		vali_accuracy_value = np.mean(np.array(accuracy_list))

		return vali_loss_value, vali_accuracy_value


def main(argv=None):
	os.environ["CUDA_VISIBLE_DEVICES"]="3" 
	tf_config=tf.ConfigProto() 
	tf_config.gpu_options.allow_growth=True 
	tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
	img_size = (32,32,3)

	model = 'rescale_fc_slim'
	learning_rate = 1e-4
	#run_name = 'rescale_label'
	option = 1
	weight_decay = 100
	print(option)

	if option == 1:
		run_name = '0.997'
		run_id = '{}_{}_lr_{}_wd_{}_{}'.format(model, run_name, learning_rate, weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))


		# First Training
		train = Train(model, img_size, learning_rate, run_id, tf_config, weight_decay)
		train.train()

	elif option == 2:
		# Filter pruning
		trained_path = '/home/spc/Dropbox/Filter_Prune/log/rescale_fc_potential_lr_0.0001_wd_100_Sep_20_14_49/model/vali_0.853-3900'
		layers = getattr(Model, model+'_variable_of_interest')()
		sel_layer_names = layers[0:len(layers)-2]
		run_name = 'Prune_Test'
		run_id = '{}_{}_lr_{}_wd_{}_{}'.format(model, run_name, learning_rate, weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))


		dict_widx, pruned_model_path = apply_pruning_scale(sel_layer_names, trained_path, run_id, tf_config)
		
		train = Train(model, img_size, learning_rate, run_id, tf_config, weight_decay)
		train.train(dict_widx=dict_widx, pruned_model_path=pruned_model_path)


if __name__ == '__main__':
	tf.app.run()
