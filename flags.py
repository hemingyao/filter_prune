import tensorflow as tf

############################################################################
#Dataset
option = 1
IMG_SIZE = (512, 512, 1)#IMG_SIZE = (32, 32, 3)
DATA_AUG = [] #DATA_AUG = ['crop', 'random_flip_left_right'] # rotation
TEST_RANGE = []
VAL_RANGE = set(range(0, 62, 5))
TRAIN_RANGE = set(range(0, 62)) - VAL_RANGE

#VAL_RANGE = set(range(0,1))
#TRAIN_RANGE = set(range(1,2)) 

#TRAIN_RANGE = ['008', '009','012', '013', '015', '020', '023',\
#         '024','031', '032', '033', '034', '035', '036']
#TRAIN = [ '004','022','026','030']
#VAL_RANGE = ['001', '002', '005', '006',]

#VAL_RANGE = set(range(0,33, 10))

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/home/spc/Documents/TFrecord/', #'./Data/'
	"""The directory of dataset""")

tf.app.flags.DEFINE_string('set_id', 'Hematoma', #cifar_tfrecord_v2
	"""The index of channel that will be used""") 

tf.app.flags.DEFINE_integer('num_labels', 2, #5000
	"""The number of images used for training""")

tf.app.flags.DEFINE_boolean('seg', True,
	"""Keep probability""")

############################################################################
# Network
tf.app.flags.DEFINE_float('learning_rate', 0.01,
	"""Keep probability""")

tf.app.flags.DEFINE_float('weight_decay', 0.001,
	"""Keep probability""")

tf.app.flags.DEFINE_float('weight_scale', 0,
	"""Keep probability""")

tf.app.flags.DEFINE_float('keep_prob_fc', 1,
	"""Keep probability""")

tf.app.flags.DEFINE_float('keep_prob_conv', 1,
	"""Keep probability""")

tf.app.flags.DEFINE_integer('batch_size', 10, 
	"""Batch size""")

tf.app.flags.DEFINE_integer('val_batch_size', 1, 
	"""Batch size""")

tf.app.flags.DEFINE_integer('num_gpus', 2,
	"""Train epoch""")

tf.app.flags.DEFINE_integer('train_epoch', 100,
	"""Train epoch""")

tf.app.flags.DEFINE_string('optimizer', 'Adam', #Momentum
	"""Train epoch""")

tf.app.flags.DEFINE_string('run_name', 'test',
	"""Train epoch""")

tf.app.flags.DEFINE_string('loss_type', 'softmax_cross_entropy', #softmax_cross_entropy, dice
	"""Train epoch""")

tf.app.flags.DEFINE_float('clip_gradients', 5.0,
	"""Keep probability""")

tf.app.flags.DEFINE_string('status', 'scratch', # "scratch", "transfer", "tune"
	"""Keep probability""")

tf.app.flags.DEFINE_string('checkpoint_file', '../Output/model3_add_loss_test/model/epoch_6.5_dice_-0.755-3900',
	"""Train epoch""")
############################################################################
# Data
tf.app.flags.DEFINE_string('net_name', 'mergeon_fourier_deep',#'model3_add_loss_more_2', 'model3_depthwise'， 'baseline_rescale'
	"""The name of the network""")

tf.app.flags.DEFINE_float('min_fraction_of_examples_in_queue', 0.5,
	"""Minimul fraction of examples in queue. Used for shuffling data""")

tf.app.flags.DEFINE_integer('num_train_images', 2000, #50000
	"""The number of images used for training""")

tf.app.flags.DEFINE_integer('num_val_images', 400, # 10000
	"""The number of images used for validation""")

tf.app.flags.DEFINE_integer('num_test_images', 400, # 16000
	"""The number of images used for validation""")

############################################################################
# Utils
tf.app.flags.DEFINE_string('log_dir', './Logs/',
	"""The diretory for logs""")
#tf.app.flags.DEFINE_integer('img_size', 256,
#	"""Image width and height""")