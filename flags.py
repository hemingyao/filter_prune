"""
Author: Heming Yao
System: Linux
"""

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

############################################################################
# Experimental Configuration
option = 1 # 1: training, 2: test, 3: filter prune

IMG_SIZE = (32, 32, 3)

DATA_AUG = ['crop', 'random_flip_left_right'] 

TEST_RANGE = set(range(0,1))
VAL_RANGE = set(range(0,1))
TRAIN_RANGE = set(range(1,2)) 

tf.app.flags.DEFINE_string('run_name', 'Full',
    """Name of your experiement""")

#TRAIN_RANGE = ['008', '009','012', '013', '015', '020', '023',\
#         '024','031', '032', '033', '034', '035', '036']
#TRAIN = [ '004','022','026','030']
#VAL_RANGE = ['001', '002', '005', '006',]


tf.app.flags.DEFINE_string('status', 'scratch', # "scratch", "transfer", "prune"
    """Keep probability""")

tf.app.flags.DEFINE_integer('save_epoch', 10,
    """Save the model at every x epoches""")

############################################################################
"""
Hyper parameters.
You need tuning them to find the best combination for your task
using cross validation
"""

tf.app.flags.DEFINE_string('net_name', 'baseline_rescale',#'model3_add_loss_more_2', 'model3_depthwise'， 'baseline_rescale'
    """The name of the network""")

tf.app.flags.DEFINE_float('learning_rate', 0.001,
    """Learning rate""")

tf.app.flags.DEFINE_float('decay_rate', 0.1,
    """Decay rate for the learning rate""")

tf.app.flags.DEFINE_integer('decay_steps', 8000,
    """Decay step for the learning rate""")

tf.app.flags.DEFINE_float('weight_decay', 0.001,
    """L2 weight decay rate""")

tf.app.flags.DEFINE_float('weight_scale', 0,
    """Only for filter pruning""")

tf.app.flags.DEFINE_float('keep_prob_fc', 0.5, #0.6
    """Keep probability for fully connected layer""")

tf.app.flags.DEFINE_float('keep_prob_conv', 0.6, #0.8
    """Keep probability for convolutional neural layer""")

tf.app.flags.DEFINE_integer('batch_size', 256*2, 
    """Batch size""")

tf.app.flags.DEFINE_integer('test_batch_size', 256, 
    """Batch size""")

tf.app.flags.DEFINE_integer('train_epoch', 500,
    """The number of training epoches""")

tf.app.flags.DEFINE_string('optimizer', 'Adam', #Momentum
    """The optimizer""")

tf.app.flags.DEFINE_string('loss_type', 'softmax_cross_entropy', #softmax_cross_entropy, dice
    """Loss function""")

tf.app.flags.DEFINE_float('clip_gradients', 0,
    """Clip gradients""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
    """The number of gpus used for training""")

tf.app.flags.DEFINE_boolean('seg', False,
    """Whether it is a segmentation task""")

tf.app.flags.DEFINE_string('checkpoint_path', '/home/spc/Dropbox/Filter_Prune/Logs/baseline_rescale_test_wd_0.0_Jul_05_15_06/model/epoch_1.0_acc_0.715-485',
    """The path for trained model for transfer learning""")

############################################################################
# Data
tf.app.flags.DEFINE_float('min_fraction_of_examples_in_queue', 0.5,
    """Minimul fraction of examples in queue. Used for shuffling data""")

tf.app.flags.DEFINE_integer('num_train_images', 50000, #50000
    """The number of images used for training""")

tf.app.flags.DEFINE_integer('num_val_images', 10000, # 10000
    """The number of images used for validation""")

tf.app.flags.DEFINE_integer('num_test_images', 1000, # 16000
    """The number of images used for test""")

tf.app.flags.DEFINE_string('save_root_for_prediction', '/home/spc/Documents/Small_Dataset/LV/', #'./Data/'
    """The directory of results should be saved""")

tf.app.flags.DEFINE_string('data_dir', '/media/DensoML/DENSO ML/tfrecord/', #'./Data/'
    """The directory of dataset""")

tf.app.flags.DEFINE_string('set_id', 'cifar_tfrecord', #cifar_tfrecord_v2
    """Specify the dataset used for training and test""") 

tf.app.flags.DEFINE_integer('num_labels', 10, #5000
    """The number of images used for training""")
############################################################################
# Utils
tf.app.flags.DEFINE_string('log_dir', './Logs/',
    """The diretory for logs""")
#tf.app.flags.DEFINE_integer('img_size', 256,
#   """Image width and height""")