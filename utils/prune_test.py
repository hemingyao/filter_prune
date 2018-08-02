import tensorflow as tf
import numpy as np
import sys, cv2, os, pickle
from flags import FLAGS
import scipy.io

def inner_fc(norm, W1, b1, W2, b2):
    out = np.matmul(norm, W1) + b1
    out = (abs(out) + out) / 2
    out = np.matmul(out, W2) + b2
    scale = 1 / (1 + np.exp(-out))
    return scale


def _filter_prune_sparse(name, weights, random=False, **kwargs):
	weights_new = np.copy(weights)

	if random == True:
		weights = np.random.normal(0,1, weights.shape)

	if 'percentage' in kwargs.keys():
		threshold = np.percentile(abs(weights), kwargs['percentage']*100)
	elif 'threshold' not in kwargs.keys():
		sys.exit('Error!')
	else:
		threshold = kwargs['threshold']

	under_threshold = abs(weights) < threshold
	weights_new[under_threshold] = 0
	drop_percent = 100*np.sum(under_threshold)/len(under_threshold.reshape(-1))

	print('{}: {:.2f} percent weights are dropped. Random = {}'.format(name, drop_percent, random))
	return weights_new, ~under_threshold


def _filter_prune_n1_channel(name, weights, random, **kwargs):
	weights_new = np.copy(weights)

	if random == True:
		weights = np.random.normal(0,1, weights.shape)

	_, _, num_channel, num_filter = weights.shape
	ls = []
	for channel in range(num_channel):
	    for out in range(num_filter):
	        weight = weights[:,:,channel, out]
	        l = np.linalg.norm(weight,ord=1)
	        ls.append(l)
	ls = np.array(ls).reshape(num_channel, num_filter)

	if 'percentage' in kwargs.keys():
		threshold = np.percentile(abs(ls), kwargs['percentage']*100)
	elif 'threshold' not in kwargs.keys():
		sys.exit('Error!')
	else:
		threshold = kwargs['threshold']

	under_threshold = abs(ls) < threshold
	under_threshold_elem = np.zeros(weights.shape, dtype=bool)
	under_threshold_elem[:,:,under_threshold] = True
	weights_new[under_threshold_elem] = 0

	drop_percent = 100*np.sum(under_threshold)/len(under_threshold.reshape(-1))

	print('{}: {:.2f} percent weights are dropped. Random = {}'.format(name, drop_percent, random))

	return weights_new, ~under_threshold_elem


def _filter_prune_n1_filter(name, weight, random, **kwargs):
	weight_new = np.copy(weight)
	c = len(weight_new.shape)

	if random == True:
		weight = np.random.normal(0,1, weight.shape)

	num_filter = weight_new.shape[-1]
	ls = []

	for out in range(num_filter):
		if c==4:
			each_filter = weight[:,:,:,out]
		elif c==5:
			each_filter = weight[:,:,:,:,out]

		temp = each_filter.reshape([-1])
		l = np.linalg.norm(temp,ord=1)
		ls.append(l)
	ls = np.array(ls)

	if 'percentage' in kwargs.keys():
		threshold = np.percentile(abs(ls), kwargs['percentage'])

	elif 'threshold' not in kwargs.keys():
		sys.exit('Error!')
	else:
		threshold = kwargs['threshold']

	under_threshold = abs(ls) < threshold
	under_threshold_elem = np.zeros(weight.shape, dtype=bool)
	if c==4:
		under_threshold_elem[:,:,:,under_threshold] = True
	elif c==5:
		under_threshold_elem[:,:,:,:,under_threshold] = True
	weight_new[under_threshold_elem] = 0

	drop_percent = 100*np.sum(under_threshold)/len(under_threshold.reshape(-1))

	print('{}: {:.2f} percent weights are dropped. Random = {}'.format(name, drop_percent, random))

	return weight_new, ~under_threshold_elem


def _filter_prune_scale(name, weight, scale_fc_W1, scale_fc_b1, scale_fc_W2, scale_fc_b2):
	weight_new = np.copy(weight)
	c = len(weight_new.shape)
	num_filter = weight.shape[-1]
	ls = []

	for out in range(0,num_filter):
		if c==4:
			each_filter = weight[:,:,:,out]
		elif c==5:
			each_filter = weight[:,:,:,:,out]
		l = np.linalg.norm(each_filter.reshape(-1),ord=1)
		ls.append(l)
	scale = inner_fc(ls, scale_fc_W1, scale_fc_b1, scale_fc_W2, scale_fc_b2)
	scale = np.array(scale)

	keep = abs(scale)>0.5
	if c==4:
		weight_new = weight_new[:,:,:,keep]
	elif c==5:
		weight_new = weight_new[:,:,:,:,keep]
	keep_percent = 100*np.sum(keep)/num_filter

	print('{}: {:.2f} percent weights are keeped. '.format(name, keep_percent))

	return weight_new, np.sum(keep)


def apply_pruning_scale(layer_names, trained_path, model_id, tf_config):
	sess = tf.Session(config=tf_config)
	nb_filters = []
	with sess.as_default():
		saver = tf.train.import_meta_graph(trained_path+'.meta')
		saver.restore(sess, trained_path)
		print('Model restored from ', trained_path)

		for i in range(len(layer_names)):
			layer_name = layer_names[i]
			var_name = layer_name+'/W:0'
			var = [v for v in tf.global_variables() if v.name == var_name][0]
			weight = var.eval()

			scale_fc_W1 = [v for v in tf.global_variables() if v.name == layer_name+'/scale/fc1/W:0'][0]
			scale_fc_b1 = [v for v in tf.global_variables() if v.name == layer_name+'/scale/fc1/b:0'][0]
			scale_fc_W2 = [v for v in tf.global_variables() if v.name == layer_name+'/scale/fc2/W:0'][0]
			scale_fc_b2 = [v for v in tf.global_variables() if v.name == layer_name+'/scale/fc2/b:0'][0]

			weight, nfilter = _filter_prune_scale(layer_name, weight, 
						scale_fc_W1.eval(), scale_fc_b1.eval(), scale_fc_W2.eval(), scale_fc_b2.eval())
			nb_filters.append(nfilter)
			# Assign new value
			sess.run(tf.assign(var, weight, validate_shape=False))
			print(var.shape, weight.shape)


		#saver = tf.train.Saver(tf.global_variables())
		if not os.path.isdir(os.path.join(FLAGS.log_dir, 'prune_model')):
			os.mkdir(os.path.join(FLAGS.log_dir, 'prune_model'))
		checkpoint_path = os.path.join(FLAGS.log_dir, 'prune_model', '{}'.format(model_id))
		#print(tf.global_variables())
		saver = tf.train.Saver(tf.global_variables())
		saver.save(sess, checkpoint_path)
	return nb_filters, checkpoint_path


def apply_pruning_random(layer_names, percents, trained_path, model_id, tf_config, random):
	assert len(layer_names)==len(percents)

	sess = tf.Session(config=tf_config)
	dict_widx = {}
	with sess.as_default():
		saver = tf.train.import_meta_graph(trained_path+'.meta')
		saver.restore(sess, trained_path)
		print('Model restored from ', trained_path)

		for i in range(len(layer_names)):
			layer_name = layer_names[i]
			var_name = layer_name+'/W:0'
			var = [v for v in tf.global_variables() if v.name == var_name][0]

			#var = [v for v in tf.global_variables() if v.name == var_name][0]
			weight = var.eval()
			print(weight.shape)
			weight, widx = _filter_prune_n1_filter(layer_names[i], weight, random=random, percentage=percents[i])

			dict_widx[var_name] = widx
			# Assign new value
			sess.run(var.assign(weight))

		#saver = tf.train.Saver(tf.global_variables())
		if not os.path.isdir(os.path.join(FLAGS.log_dir, 'prune_model')):
			os.mkdir(os.path.join(FLAGS.log_dir, 'prune_model'))
		checkpoint_path = os.path.join(FLAGS.log_dir, 'prune_model', '{}'.format(model_id))
		saver.save(sess, checkpoint_path)
	return dict_widx, checkpoint_path


def apply_prune_on_grads(grads_and_vars, dict_widx):
	for key, widx in dict_widx.items():
		count = 0
		for grad, var in grads_and_vars:
			if var.name == key:
				index = tf.cast(tf.constant(widx), tf.float32)
				grads_and_vars[count] = (tf.multiply(index, grad), var)
			count += 1
	return grads_and_vars