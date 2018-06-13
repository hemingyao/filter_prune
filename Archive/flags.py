import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('train_batch_size', 512,
	"""Train batch size""")

tf.app.flags.DEFINE_integer('test_batch_size', 1028,
	"""Test batch size""")

tf.app.flags.DEFINE_integer('train_epoch', 100,
	"""Train epoch""")

tf.app.flags.DEFINE_string('log_dir', './log/',
	"""The diretory for logs""")

tf.app.flags.DEFINE_string('test_ckpt_path', 'tflearn_model',
	"""The path of the model to be tested""")


##########################################################################

tf.app.flags.DEFINE_string('data_dir', './cifar-10-batches-bin',
	"""The directory of dataset""")

tf.app.flags.DEFINE_string('nbins', 4,
	"""The number of bins in training data""")

tf.app.flags.DEFINE_integer('num_train_images', 40000,
	"""The number of images used for training""")

tf.app.flags.DEFINE_integer('num_eval_images', 10000,
	"""The number of images used for validation""")

tf.app.flags.DEFINE_integer('num_test_images', 50000,
	"""The number of images used for test""")

tf.app.flags.DEFINE_integer('num_categories', 10,
	"""The number of categories in dataset""")

tf.app.flags.DEFINE_integer('label_bytes', 1,
	"""Label bytes""")

tf.app.flags.DEFINE_float('min_fraction_of_examples_in_queue', 0.5,
	"""Minimul fraction of examples in queue. Used for shuffling data""")







'''
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_string('train_dir', './Checkpoint/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


tf.app.flags.DEFINE_integer('validation_batch_size', 64,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_string('data_dir', './Data/Set',
                           """Path to the CIFAR-10 data directory.""")

tf.app.flags.DEFINE_integer('report_freq', 1000, 
                          """Steps takes to output errors on the screen and write summaries""")

tf.app.flags.DEFINE_integer('step_freq', 50, 
                          """Steps takes to output errors on the screen and write summaries""")

'''