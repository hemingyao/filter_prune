import tensorflow as tf
import numpy as np
import sys, cv2, os, pickle
from flags import FLAGS
import scipy.io

_EPSILON = 1e-8


def weighted_cross_entropy(y_preds, y_trues):
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


def save_tofile(Patient_No, epi, wall, Input, Label, Prediction, Location,
                                SliceSpacing, platform='python'):
    #filename = '/home/spc/Documents/Prediction/' + '{}_{:.3f}_{:.3f}'.format(
    #    			Patient_No, epi, wall)
    dirname = '/media/DensoML/DENSO ML/LVData/Prediction/LV2011/'
    if not os.path.isdir(dirname):
    	os.mkdir(dirname)

    filename = dirname + '{}_{:.3f}_{:.3f}'.format(
        			Patient_No, epi, wall)
    
    result = {'Input': Input,
			 'Label': Label,
			 'Prediction': Prediction,
              'Location': Location,
              'SliceSpacing': SliceSpacing}
    if platform=='python':
        with open(filename+'.pkl', 'wb') as handle:
            pickle.dump(result, handle)
    elif platform=='matlab':
        	scipy.io.savemat(filename, result)



def dice(y_pred, y_true, whole=False):
	y_true = y_true.astype(np.float)
	union = y_pred+y_true
	if whole:
		dice = 2*np.sum(y_pred*y_true, (0, 1, 2))/(np.sum(union, (0, 1,2))+0.00001)
	else:
		dice = 2*np.sum(y_pred*y_true, (1, 2))/(np.sum(union, (1, 2))+0.00001)
	return np.mean(dice)


def jaccard(y_pred, y_true, all=False):
	y_true = y_true.astype(np.float)
	union = y_pred+y_true - y_pred*y_true
	if all:
		jaccard = np.sum(y_pred*y_true, (0,1,2))/(np.sum(union, (0,1,2))+0.00001)
	else:
		jaccard = np.sum(y_pred*y_true, (1,2))/(np.sum(union, (1,2))+0.00001)
	return np.mean(jaccard)



def draw_wall_compare(Patient_No, wall, Input, Label, Prediction):
	num = Input.shape[0]			
	for i in range(num):
		path = '/home/spc/Documents/Prediction_cross_entro/' + '{}_{:.3f}'.format(
			Patient_No, wall)
		if not os.path.isdir(path):
			os.mkdir(path)
		print(path)

		#print(Input.shape)
		j = jaccard(Prediction[i:i+1,:,:,1], Label[i:i+1,:,:,1])
		concat_img= np.hstack((Input[i,:,:]*255, 
			Label[i,:,:,1]*255, 
			Prediction[i,:,:,1]*255))

		name =  path + '/' + str(i)
		cv2.imwrite('{}_{:.3f}.png'.format(name, j), concat_img)


def draw_prediction_compare(Patient_No, epi, wall, Input, Label, Prediction, Location=None):
	num = Input.shape[0]			
	root = '/home/spc/Documents/Prediction_Sunny/'
	if not os.path.isdir(root):
		os.mkdir(root)
	for i in range(num):
		path = root + '{}_{:.3f}_{:.3f}'.format(
			Patient_No, epi, wall)
		if not os.path.isdir(path):
			os.mkdir(path)

		#print(Input.shape)
		j = jaccard(Prediction[i:i+1,:,:,1], Label[i:i+1,:,:,1])
		concat_img= np.hstack((Input[i,:,:]*255, 
			(Label[i,:,:,1]*0.5+Label[i,:,:,2])*255, 
			(Prediction[i,:,:,1]*0.5+Prediction[i,:,:,2])*255))

		name =  path + '/' + str(i)
		cv2.imwrite('{}_{:.3f}_{}_{}.png'.format(name, j, int(Location[i][1]), int(Location[i][2])), concat_img)


def draw_prediction_compare_multi(Patient_No, Input, Label, Prediction):
	num = Input.shape[0]			
	root = '/home/spc/Documents/Multi-output/'
	if not os.path.isdir(root):
		os.mkdir(root)
	for i in range(num):
		path = root + '{}'.format(
			Patient_No)
		if not os.path.isdir(path):
			os.mkdir(path)


		j_1 = jaccard(Prediction[i:i+1,:,0:256,1], Label[i:i+1,:,:,1])
		j_2 = jaccard(Prediction[i:i+1,:,256*1:256*2,1], Label[i:i+1,:,:,1])
		j_3 = jaccard(Prediction[i:i+1,:,256*2:256*3,1], Label[i:i+1,:,:,1])
		j_4 = jaccard(Prediction[i:i+1,:,256*3:256*4,1], Label[i:i+1,:,:,1])
		j_5 = jaccard(Prediction[i:i+1,:,256*4:256*5,1], Label[i:i+1,:,:,1])

		concat_img= np.hstack((Input[i,:,:]*255, 
			(Label[i,:,:,1]*0.5+Label[i,:,:,2])*255, 
			(Prediction[i,:,:,1]*0.5+Prediction[i,:,:,2])*255))

		name =  path + '/' + str(i)
		cv2.imwrite('{}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}.png'.format(name, j_1, j_2, j_3, j_4, j_5), concat_img)



def samplewise_zero_center(batch, per_channel=False):
	for i in range(len(batch)):
		if not per_channel:
			batch[i] -= np.mean(batch[i])
		else:
			batch[i] -= np.mean(batch[i], axis=(0, 1, 2), keepdims=True)
	return batch


def samplewise_stdnorm(batch):
	for i in range(len(batch)):
		batch[i] /= (np.std(batch[i], axis=0) + _EPSILON)
	return batch


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 


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


def _filter_prune_n1_filter(name, weights, random, **kwargs):
	weights_new = np.copy(weights)

	if random == True:
		weights = np.random.normal(0,1, weights.shape)

	_, _, _, num_channel, num_filter = weights.shape
	ls = []

	for out in range(num_filter):
		weight = weights[:,:,:,:, out]
		weight = weight.reshape([-1])
		l = np.linalg.norm(weight,ord=1)
		ls.append(l)
	ls = np.array(ls)


	if 'percentage' in kwargs.keys():
		threshold = np.percentile(abs(ls), kwargs['percentage'])
	elif 'threshold' not in kwargs.keys():
		sys.exit('Error!')
	else:
		threshold = kwargs['threshold']

	under_threshold = abs(ls) < threshold
	under_threshold_elem = np.zeros(weights.shape, dtype=bool)
	under_threshold_elem[:,:,:,:,under_threshold] = True
	weights_new[under_threshold_elem] = 0

	drop_percent = 100*np.sum(under_threshold)/len(under_threshold.reshape(-1))

	print('{}: {:.2f} percent weights are dropped. Random = {}'.format(name, drop_percent, random))

	return weights_new, ~under_threshold_elem


def _filter_prune_scale(name, weight, scale_fc_W1, scale_fc_b1, scale_fc_W2, scale_fc_b2):
	weight_new = np.copy(weight)

	num_filter = weight.shape[-1]
	ls = []
	for out in range(0,num_filter):
	    each_filter = weight[:,:,:,:,out]
	    l = np.linalg.norm(each_filter.reshape(-1),ord=1)
	    ls.append(l)
	scale = inner_fc(ls, scale_fc_W1, scale_fc_b1, scale_fc_W2, scale_fc_b2)
	scale = np.array(scale)

	under_threshold = abs(scale) < 0.50
	under_threshold_elem = np.zeros(weight.shape, dtype=bool)
	under_threshold_elem[:,:,:,:,under_threshold] = True
	weight_new[under_threshold_elem] = 0

	drop_percent = 100*np.sum(under_threshold)/len(under_threshold.reshape(-1))

	print('{}: {:.2f} percent weights are dropped. '.format(name, drop_percent))

	return weight_new, ~under_threshold_elem


def apply_pruning_scale(layer_names, trained_path, model_id, tf_config):
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
			weight = var.eval()

			scale_fc_W1 = [v for v in tf.global_variables() if v.name == layer_name+'/scale/fc1/W:0'][0]
			scale_fc_b1 = [v for v in tf.global_variables() if v.name == layer_name+'/scale/fc1/b:0'][0]
			scale_fc_W2 = [v for v in tf.global_variables() if v.name == layer_name+'/scale/fc2/W:0'][0]
			scale_fc_b2 = [v for v in tf.global_variables() if v.name == layer_name+'/scale/fc2/b:0'][0]

			weight, widx = _filter_prune_scale(layer_name, weight, 
						scale_fc_W1.eval(), scale_fc_b1.eval(), scale_fc_W2.eval(), scale_fc_b2.eval())

			dict_widx[var_name] = widx
			# Assign new value
			sess.run(var.assign(weight))

		#saver = tf.train.Saver(tf.global_variables())
		if not os.path.isdir(os.path.join(FLAGS.log_dir, 'prune_model')):
			os.mkdir(os.path.join(FLAGS.log_dir, 'prune_model'))
		checkpoint_path = os.path.join(FLAGS.log_dir, 'prune_model', '{}'.format(model_id))
		saver.save(sess, checkpoint_path)
	return dict_widx, checkpoint_path


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


def calculate_number_of_parameters(variables):
	total_parameters = 0
	for variable in variables:
		print(variable)
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
		print(variable_parameters)
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

	