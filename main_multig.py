
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from scipy.misc import imresize
#import keras.backend as K
import numpy as np
import time, os, h5py
import functools, itertools, six
from six.moves import xrange

from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops

## Import Customized Functions
import network
from flags import FLAGS
from utils import *
from train_ops import *
from batch_generator import ReadData
import tflearn_dev as tflearn
from data_flow import input_fn
import multig_utils 

_FLOATX = tf.float32
_EPSILON = 1e-10

VAL_RANGE = set(range(0, 1))
TRAIN_RANGE = set(range(1, 6)) 

#VAL_RANGE = set(range(0,33, 10))

option = 1

RUN_NAME = '10times'

class Train():
    def __init__(self, run_id, config, img_size):
        self.img_size = img_size
        self.run_id = run_id
        self.dict_widx = None
        self.tf_config = config

        # From FLAGS
        self.train_range = TRAIN_RANGE
        self.vali_range = VAL_RANGE
        self.wd = FLAGS.weight_decay
        self.wd_scale = FLAGS.weight_scale
        self.set_id = FLAGS.set_id
        self.am_training = True
        

    def _tower_fn(self, index):
        """
        Build computation tower 
        """
        logits = getattr(network, FLAGS.net_name)(inputs=self.batch_data[index], 
            prob_fc=self.prob_fc, prob_conv=self.prob_conv, 
            wd=self.wd, wd_scale=self.wd_scale, 
            training_phase=self.am_training)

        tower_pred = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits)
            }

        tower_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.batch_labels[index], logits=logits, name='cross_entropy'))

        model_params = tf.trainable_variables()
        tower_total_loss = add_all_losses(tower_loss)

        tower_grad = tf.gradients(tower_total_loss, model_params)

        return tower_loss, tower_total_loss, zip(tower_grad, model_params), tower_pred


    def _build_graph(self):
        """Resnet model body.
        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
           manages gradient updates.
        Args:
          features: a list of tensors, one for each tower
          labels: a list of tensors, one for each tower
          mode: ModeKeys.TRAIN or EVAL
          params: Hyperparameters suitable for tuning
        Returns:
          A EstimatorSpec object.
        """
        global_step = tf.contrib.framework.get_or_create_global_step()
        tower_losses = []
        tower_total_losses = []
        tower_gradvars = []
        tower_preds = []
        num_gpus = FLAGS.num_gpus
        variable_strategy = 'GPU'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = multig_utils.local_device_setter(
                  worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = multig_utils.local_device_setter(
                  ps_device_type='gpu',
                  worker_device=worker_device,
                  ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                      num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope(FLAGS.net_name, reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, total_loss, gradvars, preds = self._tower_fn(i)
                        tower_losses.append(loss)
                        tower_total_losses.append(total_loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                        # Only trigger batch_norm moving mean and variance update from
                        # the 1st tower. Ideally, we should grab the updates from all
                        # towers but these stats accumulate extremely fast so we can
                        # ignore the other stats from the other towers without
                        # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                       name_scope)
        # Now compute global loss and gradients.
        gradlst = []
        varlst = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradlst.append(avg_grad)
                varlst.append(var)

                if FLAGS.clip_gradients > 0.0:
                    gradlst, grad_norm = tf.clip_by_global_norm(gradlst, FLAGS.clip_gradients)
                gradvars = list(zip(gradlst, varlst))


        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        

        with tf.device(consolidation_device):
          # Suggested learning rate scheduling from

            self.loss = tf.reduce_mean(tower_losses, name='loss')
            self.total_loss = tf.reduce_mean(tower_total_losses, name='total_loss')

            _, apply_gradients_op, learning_rate = train_operation(lr=FLAGS.learning_rate, global_step=global_step, 
                            decay_rate=0.5, decay_steps=32000, optimizer=FLAGS.optimizer, clip_gradients=FLAGS.clip_gradients,
                            loss=self.total_loss, var_list=tf.trainable_variables(), grads_and_vars=gradvars)

            # log
            examples_sec_hook = multig_utils.ExamplesPerSecondHook(
                FLAGS.batch_size, every_n_steps=100)

            tensors_to_log = {'learning_rate': learning_rate, 'loss': self.loss}

            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)

            train_hooks = [logging_hook, examples_sec_hook]

            """
            if True:  ##TODO true or false
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, replicas_to_aggregate=num_workers)
                sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
                train_hooks.append(sync_replicas_hook)
            """

            predictions = {
                'classes':
                    tf.concat([p['classes'] for p in tower_preds], axis=0),
                'probabilities':
                    tf.concat([p['probabilities'] for p in tower_preds], axis=0)
            }
            stacked_labels = tf.concat(self.batch_labels, axis=0)
            #stacked_labels = tf.argmax(input=stacked_labels, axis=1),
            correct_prediction = tf.equal(predictions['classes'], tf.argmax(stacked_labels,1))

            metrics = {
                'accuracy': tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    #tf.metrics.accuracy(stacked_labels, predictions['classes'])
            }



            # Create single grouped train op
            train_op = [apply_gradients_op]
            train_op.extend(update_ops)
            self.train_op = tf.group(*train_op)
            
            self.prediction = predictions['classes']
            self.accuracy = metrics['accuracy']

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        self.summary_op = tf.summary.merge_all()

    def train(self, **kwargs):
        #with tf.Graph().as_default():
        ops.reset_default_graph()
        sess = tf.Session(config=self.tf_config)

        with sess.as_default():
            # Data Reading objects
            tflearn.is_training(True, session=sess)

            self.am_training = tf.placeholder(dtype=bool, shape=())
            self.prob_fc = tf.placeholder_with_default(0.5, shape=())
            self.prob_conv = tf.placeholder_with_default(0.5, shape=())


            data_fn = functools.partial(input_fn, data_dir=os.path.join(FLAGS.data_dir, FLAGS.set_id), 
                num_shards=FLAGS.num_gpus, batch_size=FLAGS.batch_size, use_distortion_for_training=True)

            self.batch_data, self.batch_labels = tf.cond(self.am_training, 
                lambda: data_fn(subset='train'),
                lambda: data_fn(subset='test'))

            #self.batch_data = self.batch_data[0]
            #self.batch_labels = self.batch_labels[0]

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

                #tflearn.config.init_training_mode()
                self._build_graph()             
                #tflearn.config.init_training_mode()
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


            train_steps_per_epoch = FLAGS.num_train_images//FLAGS.batch_size
            report_freq = train_steps_per_epoch

            train_steps = FLAGS.train_epoch * train_steps_per_epoch

            durations = []
            train_loss_list = []
            train_total_loss_list = []
            train_accuracy_list = []
            best_epoch = 0
            best_accuracy = 0
            best_loss = 1

            nparams = calculate_number_of_parameters(tf.trainable_variables())
            print(nparams)


            for step in range(train_steps):

                #print('{} step starts'.format(step))
                start_time = time.time()
                tflearn.is_training(True, session=sess)

                data, labels, _, summary_str, loss_value, total_loss, accuracy = sess.run(
                    [self.batch_data, self.batch_labels, self.train_op, self.summary_op, self.loss, self.total_loss, self.accuracy], 
                    feed_dict={self.am_training: True, self.prob_fc: FLAGS.keep_prob_fc, self.prob_conv: FLAGS.keep_prob_conv})

                tflearn.is_training(False, session=sess)
                duration = time.time() - start_time
                #print('{} step starts {}'.format(step, duration))
                
                durations.append(duration)
                train_loss_list.append(loss_value)
                train_total_loss_list.append(total_loss)
                train_accuracy_list.append(accuracy)

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                
                #if step%(report_freq*10)==0:
                #   print(self.learning_rate)
                if step%report_freq == 0:
                    start_time = time.time()

                    summary_writer.add_summary(summary_str, step)

                    sec_per_report = np.sum(np.array(durations))
                    train_loss = np.mean(np.array(train_loss_list))
                    train_total_loss = np.mean(np.array(train_total_loss_list))
                    train_accuracy_value = np.mean(np.array(train_accuracy_list))

                    train_loss_list = []
                    train_total_loss_list = []
                    train_accuracy_list = []
                    durations = []

                    train_summ = tf.Summary()
                    train_summ.value.add(tag="train_loss", simple_value=train_loss.astype(np.float))
                    train_summ.value.add(tag="train_total_loss", simple_value=train_total_loss.astype(np.float))
                    train_summ.value.add(tag="train_accuracy", simple_value=train_accuracy_value.astype(np.float))

                    summary_writer.add_summary(train_summ, step)
                                                      
                    vali_loss_value, vali_accuracy_value = self._full_validation(sess)
                    
                    if step%(report_freq*50)==0:
                        epoch = step/(report_freq*10)
                        model_dir = os.path.join(FLAGS.log_dir, self.run_id, 'model')
                        if not os.path.isdir(model_dir):
                            os.mkdir(model_dir)
                        checkpoint_path = os.path.join(model_dir, 'epoch_{}_acc_{:.3f}'.format(epoch, vali_accuracy_value))

                        self.saver.save(sess, checkpoint_path, global_step=step)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag="vali_loss", simple_value=vali_loss_value.astype(np.float))
                    vali_summ.value.add(tag="vali_accuracy", simple_value=vali_accuracy_value.astype(np.float))


                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                    vali_duration = time.time() - start_time

                    format_str = ('Epoch %d, loss = %.4f, total_loss = %.4f, acc = %.4f, vali_loss = %.4f, val_acc = %.4f (%.3f ' 'sec/report)')
                    print(format_str % (step//report_freq, train_loss, train_total_loss, train_accuracy_value, vali_loss_value, vali_accuracy_value, sec_per_report+vali_duration))


    def _full_validation(self, sess):
        tflearn.is_training(False, session=sess)
        num_batches_vali = FLAGS.num_val_images // FLAGS.batch_size

        loss_list = []
        accuracy_list = []

        for step_vali in range(num_batches_vali):
            _, _, loss, accuracy = sess.run([self.batch_data, self.batch_labels,self.loss, self.accuracy], 
                feed_dict={self.am_training:False, self.prob_fc: 1, self.prob_conv: 1})
                                            #feed_dict={self.am_training: False, self.prob_fc: FLAGS.keep_prob_fc, self.prob_conv: 1})
            
            loss_list.append(loss)
            accuracy_list.append(accuracy)

        vali_loss_value = np.mean(np.array(loss_list))
        vali_accuracy_value = np.mean(np.array(accuracy_list))

        return vali_loss_value, vali_accuracy_value


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("gi", nargs='?', help="index of the gpu",
                        type=int)
    gi = parser.parse_args().gi

    os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"

    #os.environ["CUDA_VISIBLE_DEVICES"]= str(gi)
    tf_config=tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True 

    img_size = (32, 32, 3)
    #set_id = 'eval'

    if option == 1:
        run_id = '{}_{}_lr_{}_wd_{}_{}'.format(FLAGS.net_name, RUN_NAME, FLAGS.learning_rate, FLAGS.weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))

        # First Training
        train = Train(run_id, tf_config, img_size)
        train.train()

    elif option == 2:
        test_new(tf_config, img_size, label_size, test_path, test_range=VAL_RANGE)

    elif option == 3:
        # Filter pruning
        trained_path = '/home/spc/Dropbox/Drowsiness/3D/log_0.0001/Sparse_fourier_new_7__lr_0.001_wd_0.0001_Feb_06_13_47/model/epoch_11.0_dice_-0.832-6600'

        run_name = 'Prune_1'
        run_id = '{}_{}_lr_{}_wd_{}_{}'.format(model_name, run_name, learning_rate, weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))
        layer_names = ['Conv3D', 'Conv3D_1', 'Conv3D_2', 'Conv3D_3'] 

        dict_widx, pruned_model_path = apply_pruning_scale(layer_names, trained_path, run_id, tf_config)
        #dict_widx, pruned_model_path = apply_pruning_random(layer_names, [58,52,52,63], trained_path, run_id, tf_config, random=False)

        train = Train(model_name, run_id, tf_config, set_id, img_size, learning_rate, train_range, vali_range, weight_decay)
        train.train(dict_widx=dict_widx, pruned_model_path=pruned_model_path)


if __name__ == '__main__':
    tf.app.run()











