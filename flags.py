import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

############################################################################
# Network
tf.app.flags.DEFINE_float('learning_rate', 0.01,
	"""Keep probability""")

tf.app.flags.DEFINE_float('weight_decay', 0.0005,
	"""Keep probability""")

tf.app.flags.DEFINE_float('weight_scale', 0,
	"""Keep probability""")

tf.app.flags.DEFINE_float('keep_prob_fc', 0.5,
	"""Keep probability""")

tf.app.flags.DEFINE_float('keep_prob_conv', 0.5,
	"""Keep probability""")

tf.app.flags.DEFINE_integer('batch_size', 128, 
	"""Batch size""")

tf.app.flags.DEFINE_integer('val_batch_size', 128, 
	"""Batch size""")


tf.app.flags.DEFINE_integer('train_epoch', 100,
	"""Train epoch""")

tf.app.flags.DEFINE_string('optimizer', 'Momentum',
	"""Train epoch""")
############################################################################
# Data
tf.app.flags.DEFINE_string('net_name', 'baseline_rescale',#'model3_add_loss_more_2', 'model3_depthwise'， 'model3_add_loss'
	"""The name of the network""")

tf.app.flags.DEFINE_string('data_dir', './Data/',
	"""The directory of dataset""")

tf.app.flags.DEFINE_string('set_id', 'cifar_tfrecord',
	"""The index of channel that will be used""") 

tf.app.flags.DEFINE_integer('label_bytes', 1, #5000
	"""The number of images used for training""")

tf.app.flags.DEFINE_float('min_fraction_of_examples_in_queue', 0.1,
	"""Minimul fraction of examples in queue. Used for shuffling data""")

tf.app.flags.DEFINE_integer('num_train_images', 50000, #50000
	"""The number of images used for training""")

tf.app.flags.DEFINE_integer('num_val_images', 128, # 10000
	"""The number of images used for validation""")

tf.app.flags.DEFINE_integer('num_test_images', 300, # 16000
	"""The number of images used for validation""")

############################################################################
# Utils
tf.app.flags.DEFINE_string('log_dir', './output/',
	"""The diretory for logs""")
#tf.app.flags.DEFINE_integer('img_size', 256,
#	"""Image width and height""")